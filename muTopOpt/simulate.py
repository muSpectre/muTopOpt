#!/usr/bin/env python3
#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Elasticity driver of muTopOpt: optimize the density of a periodic unit cell,
in 2D or 3D, so that its homogenized stiffness matches a target isotropic
stiffness, given either as bulk and shear moduli (``target.bulk_modulus``,
``target.shear_modulus``) or as Young's modulus and Poisson's ratio
(``target.young_modulus``, ``target.poisson_ratio``).

A run is described by a JSON control file; ``--write-defaults FILE`` writes
one with every key at its default, ready to edit, together with its JSON
Schema. Keys the file does not set take their defaults; ``null`` means the
value is determined automatically (the help of each key says how).

Mathematical setup
------------------
The design is a density rho in [0, 1] -- a nodal finite-element field by
default (``discretization.density``), or one value per pixel -- mapped to
per-pixel Lame parameters by SIMP with exponent p
(``material.simp_exponent``). Each of the d(d+1)/2 load cases G applies a
macro strain E_G of magnitude 0.01 along one direction of the symmetric
strain space; the periodic displacement fluctuation solves the FE equilibrium

    K(rho) u_G = -B^T C(rho) : E_G

by FFT-preconditioned CG, and the homogenized stress <sigma_G> is compared
with the target stress C* : E_G:

    f(rho) = sum_G |<sigma_G> - C*:E_G|^2 / |C*:E_G|^2 + f_reg(rho),

where f_reg is the Modica-Mortola phase-field regularization with interface
width ``regularization.interface_width`` and weight
``regularization.weight``. The gradient comes from the discrete adjoint (one
extra solve per load case). The outer optimizer (``optimizer.method``) is
either NuMPI's bound-constrained trust-region Newton-CG with exact
Hessian-vector products or NuMPI's projected L-BFGS; both run MPI-distributed
on the same domain decomposition as the fields. The inner CG tolerance adapts
from ``inner_solver.adaptive.initial_tolerance`` down to its minimum (see
muTopOpt.optimize). The run starts from a generated field
(``initial_density.source``) or from the last frame of a previous output
file, which is Fourier-resampled if its grid differs.

The solves and the sensitivity run on the host or, with a GPU build of
muGrid and ``compute.device`` = "gpu", on the device; the outer optimizer
always runs on the host. The density, per-frame homogenized stresses and
effective moduli are written to the NetCDF file ``output.file`` through
muGrid, together with the resolved control file; it can be used as
``initial_density.source`` of a later run.

Examples
--------
    mutopopt --write-defaults run.json   # then edit run.json
    mutopopt run.json
    mpirun -np 4 mutopopt run.json

A minimal control file only sets what differs from the defaults, e.g.

    {"grid": {"nb_grid_pts": [128, 128, 128]},
     "target": {"young_modulus": 0.2, "poisson_ratio": -0.3},
     "compute": {"device": "gpu", "precision": "single"}}
"""

import argparse
import json
import os
import shlex
import sys
import time

import muGrid
import numpy as np

from muTopOpt import (
    E_nu_from_lame,
    Homogenization,
    NodalPhaseFieldRegularization,
    PhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.bounds import target_feasibility
from muTopOpt.homogenization import gpu_aware_mpi
from muTopOpt.control import (
    ControlError,
    defaults,
    describe,
    dumps,
    json_schema,
)
from muTopOpt.control import load as load_control
from muTopOpt.loadcases import (
    isotropic_stiffness_from_E_nu,
    isotropic_stiffness_tensor,
    target_load_cases,
)
from muTopOpt.optimize import (
    initial_density,
    optimize_bounded_lbfgs,
    optimize_trust_region,
)
from muTopOpt.restart import INITIAL_DENSITY_KINDS, restart_density
from muTopOpt.timing import timer
from muTopOpt.version import (
    mugrid_version,
    mutopopt_version,
    provenance_attributes,
)


#: File name of the JSON Schema that --write-defaults writes next to the
#: control file and that the control file's "$schema" key points to.
SCHEMA_FILE = "simulate.schema.json"


def _settings(control):
    """The run settings of a validated control object, under the names the
    driver uses internally."""
    c = control
    tr = c["optimizer"]["trust_region"]
    adaptive = c["inner_solver"]["adaptive"]
    method = c["optimizer"]["method"]
    return argparse.Namespace(
        nb_grid_pts=c["grid"]["nb_grid_pts"],
        domain_lengths=c["grid"]["domain_lengths"],
        element=c["discretization"]["element"],
        density=c["discretization"]["density"],
        solid_E=c["material"]["young_modulus"],
        solid_nu=c["material"]["poisson_ratio"],
        penalty=c["material"]["simp_exponent"],
        void_ratio=c["material"]["void_ratio"],
        target_K=c["target"]["bulk_modulus"],
        target_G=c["target"]["shear_modulus"],
        target_E=c["target"]["young_modulus"],
        target_nu=c["target"]["poisson_ratio"],
        eta=c["regularization"]["interface_width"],
        reg_weight=c["regularization"]["weight"],
        init=c["initial_density"]["source"],
        init_volume_fraction=c["initial_density"]["volume_fraction"],
        init_length=c["initial_density"]["correlation_length"],
        seed=c["initial_density"]["seed"],
        optimizer={None: "auto", "trust_region": "tr",
                   "lbfgs": "lbfgs"}[method],
        bfgs_maxiter=c["optimizer"]["max_iterations"],
        bfgs_gtol=c["optimizer"]["gradient_tolerance"],
        bfgs_xtol=c["optimizer"]["step_tolerance"],
        tr_delta0=tr["initial_radius"],
        tr_delta_max=tr["max_radius"],
        tr_eta=tr["acceptance_threshold"],
        hv_cg_tol=tr["hessian_tolerance"],
        hv_warm_start=tr["hessian_warm_start"],
        cg_tol=c["inner_solver"]["tolerance"],
        cg_maxiter=c["inner_solver"]["max_iterations"],
        preconditioner=c["inner_solver"]["preconditioner"],
        output_cg_iters=c["inner_solver"]["print_iterations"],
        cg_tol_start=(adaptive["initial_tolerance"] if adaptive["enabled"]
                      else 0.0),
        cg_tol_min=adaptive["minimum_tolerance"],
        cg_forcing_exp=adaptive["forcing_exponent"],
        cg_stall_shrink=adaptive["stall_shrink"],
        cg_stall_rel=adaptive["stall_relative_decrease"],
        device=c["compute"]["device"],
        precision=c["compute"]["precision"],
        output=c["output"]["file"],
        dump_every=(c["output"]["dump_every"] if c["output"]["dump_every"] > 0
                    else -1),
        no_flush=not c["output"]["flush_frames"],
    )


def _write_defaults(path):
    """Write the default control file and its JSON Schema next to it."""
    schema_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                               SCHEMA_FILE)
    control = {"$schema": SCHEMA_FILE, **defaults()}
    with open(path, "w") as f:
        f.write(dumps(control))
    with open(schema_path, "w") as f:
        f.write(json.dumps(json_schema(), indent=2) + "\n")
    print(f"wrote {path} (all keys at their defaults) and {schema_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Keys of the control file, with their defaults:\n"
        + describe(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "control", nargs="?", metavar="CONTROL.json",
        help="JSON control file; keys it does not set take their defaults")
    p.add_argument(
        "--write-defaults", metavar="FILE",
        help=f"write a control file with every key at its default (and the "
        f"JSON Schema {SCHEMA_FILE} next to it), then exit")
    cli = p.parse_args()
    if cli.write_defaults is not None:
        if muGrid.Communicator().rank == 0:
            _write_defaults(cli.write_defaults)
        return
    if cli.control is None:
        p.error("a control file is required; create one with "
                "--write-defaults FILE")
    try:
        control = load_control(cli.control)
    except (ControlError, OSError) as err:
        p.error(str(err))
    args = _settings(control)

    dim = len(args.nb_grid_pts)
    if args.domain_lengths is not None and len(args.domain_lengths) != dim:
        p.error(f"grid.domain_lengths must have {dim} values, one per entry "
                "of grid.nb_grid_pts")
    if args.init not in INITIAL_DENSITY_KINDS and not os.path.exists(args.init):
        p.error(
            f"initial_density.source must be one of "
            f"{', '.join(INITIAL_DENSITY_KINDS)} or an existing NetCDF "
            f"restart file; '{args.init}' is neither")

    # Resolve 'auto': the trust-region Newton-CG is the robust default (its
    # acceptance test cannot drown in inner-solve noise the way a line search
    # does), but it needs a NuMPI providing tr_newton_bounded.
    optimizer = args.optimizer
    if optimizer == "auto":
        try:
            from NuMPI.Optimization import tr_newton_bounded  # noqa: F401

            optimizer = "tr"
        except ImportError:
            optimizer = "lbfgs"

    # Precision-aware accuracy limits. In float32 the *true* residual b - Kx
    # of the CG solve stagnates at |r|/|b| ~ 1.5e-6 (only the recursive CG
    # residual keeps shrinking below that, and CG can fail outright), so
    # tolerances below ~1e-6 buy no true accuracy.
    #
    # The gradient tolerance is measured on the mesh-invariant volume-fraction
    # derivative ĝ = (V/V_pixel) df/drho = N df/drho (see muTopOpt.optimize.
    # _gradient_scale): the raw per-pixel gradient shrinks like 1/N with
    # resolution, so an absolute per-pixel gtol would certify an unchanged
    # initial design as converged on a fine enough grid. Typical initial
    # designs start at ||ĝ||_inf ~ O(100). For scale: on a 64^3 grid
    # (N ~ 2.6e5) ĝ = 25 corresponds to a per-pixel gradient of ~1e-4 and
    # ĝ = 2.5 to ~1e-5. The single-precision default is 10x coarser because
    # the achievable ĝ accuracy at the float32 residual floor makes a tighter
    # tolerance uncertifiable.
    # TODO: rationale for the double-precision default gtol of 2.5?
    single = args.precision == "single"
    rtol_floor = 1e-6 if single else 0.0
    bfgs_gtol = (args.bfgs_gtol if args.bfgs_gtol is not None
                 else (25.0 if single else 2.5))

    # Adaptive inner CG tolerance is on by default (disabled: the fixed
    # inner_solver.tolerance throughout). The floor defaults per optimizer:
    # L-BFGS needs the final gradient error below the gradient tolerance to
    # certify convergence -- in the mesh-invariant ĝ units the error is
    # bounded by ~1e3 x the CG tolerance (measured at 32^3/64^3 on the
    # initial iterate; _KAPPA_EFF below includes a 10x margin for late
    # high-contrast designs), so the floor is bfgs_gtol/_KAPPA_EFF, capped at
    # the fixed inner_solver.tolerance. The trust-region accuracy control only
    # tightens as far as the predicted reduction requires, so it just gets
    # ample room.
    _KAPPA_EFF = 1e4  # mesh-invariant gradient error per unit CG rtol
    cg_tol_start = (args.cg_tol_start
                    if args.cg_tol_start and args.cg_tol_start > 0 else None)
    if args.cg_tol_min is not None:
        cg_tol_min = args.cg_tol_min
        if cg_tol_min < rtol_floor:
            import warnings

            warnings.warn(
                f"inner_solver.adaptive.minimum_tolerance {cg_tol_min:.1e} "
                "is below the single-"
                f"precision solve accuracy floor (~{rtol_floor:.0e}); the "
                "true residual cannot reach it, so it will be raised to the "
                "floor",
                RuntimeWarning,
            )
    elif optimizer == "tr":
        cg_tol_min = 1e-10
    else:
        cg_tol_min = min(args.cg_tol, bfgs_gtol / _KAPPA_EFF)
    if cg_tol_start is None and optimizer == "tr":
        # Adaptation disabled: pin the trust-region accuracy control at the
        # fixed inner_solver.tolerance (start == floor), i.e. a truly fixed tolerance.
        cg_tol_start = args.cg_tol
        cg_tol_min = args.cg_tol
    # Apply the precision floor *last*, after every branch above has set
    # cg_tol_min/cg_tol_start, so that no branch (including the one that
    # disables adaptation) can bypass it.
    cg_tol_min = max(cg_tol_min, rtol_floor)
    if cg_tol_start is not None:
        cg_tol_start = max(cg_tol_start, cg_tol_min)

    # MPI_COMM_WORLD under a launcher (muGrid starts MPI itself, no mpi4py
    # needed); serial otherwise
    comm = muGrid.Communicator.world()
    rank0 = comm.rank == 0

    material = SimpMaterial(args.solid_E, args.solid_nu, args.penalty, args.void_ratio)
    # CLI exposes "single"/"double"; muTopOpt works in NumPy dtypes internally.
    dtype = {"single": np.float32, "double": np.float64}[args.precision]
    with timer("setup"):
        homog = Homogenization(
            tuple(args.nb_grid_pts),
            material,
            comm=comm,
            element=args.element,
            domain_lengths=args.domain_lengths,
            preconditioner=args.preconditioner,
            cg_tol=args.cg_tol,
            cg_maxiter=args.cg_maxiter,
            cg_verbose=args.output_cg_iters,
            device=args.device,
            dtype=dtype,
        )
    # Banner first, before the target checks and the restart read: a run that
    # dies in setup should still say which build it was. muGrid prints from
    # every rank (its banner carries the per-rank decomposition); muTopOpt's is
    # rank-independent, so rank 0 prints it alone.
    print(mugrid_version(homog.comm, homog.device))
    if rank0:
        print(mutopopt_version())
        if homog.device_memory != "host":
            line = f"device memory: {homog.device_memory_reason}"
            if homog.comm.size > 1:
                line += ("; MPI takes device buffers" if gpu_aware_mpi()
                         else "; MPI does not take device buffers, so they "
                              "are staged through the host")
            print(line)
    if args.target_E is not None:
        target = isotropic_stiffness_from_E_nu(dim, args.target_E, args.target_nu)
    else:
        target = isotropic_stiffness_tensor(dim, args.target_K, args.target_G)
    strain_magnitude = 0.01
    cases = target_load_cases(dim, target, magnitude=strain_magnitude)
    # Applied deformation gradient of each load case, F = I + macro_strain (the
    # macro strain is the applied displacement gradient). Written per frame so a
    # deformation-path simulation could vary it; for topology optimization it is
    # constant across frames.
    applied_deformation_gradient = np.stack(
        [np.eye(dim) + lc.macro_strain for lc in cases]
    )

    # Isotropic-equivalent effective bulk (K) and shear (G) moduli from the
    # per-load-case homogenized stress response to the unit strains, in the same
    # (K, G) parameterization as the target (sigma = 2 G dev(E) + K tr(E) I).
    # Used to report the current elastic properties against the target.
    def effective_moduli(stresses):
        sigma_vol = sum(stresses[i] for i in range(dim))  # response to E = m*I
        K = float(np.trace(sigma_vol)) / (dim * dim * strain_magnitude)
        shear = []
        for k in range(dim, len(cases)):  # shear load cases follow the diagonal
            ij = np.unravel_index(
                int(np.argmax(np.abs(np.triu(cases[k].macro_strain, 1)))), (dim, dim)
            )
            shear.append(float(stresses[k][ij]) / strain_magnitude)
        return K, float(np.mean(shear))

    # Young's modulus and Poisson's ratio equivalent to (K, G) in the same
    # convention (K = lambda + 2 mu / dim, G = mu; plane-strain in 2D, so these
    # are the true 3D constants, matching target.young_modulus/poisson_ratio).
    def E_nu_from_K_G(K, G):
        return E_nu_from_lame(K - 2.0 * G / dim, G)

    target_K, target_G = effective_moduli([lc.target_stress for lc in cases])
    target_E, target_nu = E_nu_from_K_G(target_K, target_G)

    # All four effective moduli of a homogenized stress response, in the order
    # they are reported and written to the output.
    _MODULI = ("K", "G", "E", "nu")

    def moduli_of(stresses):
        K, G = effective_moduli(stresses)
        E, nu = E_nu_from_K_G(K, G)
        return {"K": K, "G": G, "E": E, "nu": nu}

    targets = {"K": target_K, "G": target_G, "E": target_E, "nu": target_nu}
    # The interface width defaults to one grid spacing: narrow enough for crisp
    # designs, wide enough that the regularization can still move interfaces
    # (merge/remove features) instead of freezing the initial topology -- and
    # comfortably clear of the lattice-pinning regime (see
    # test/test_interface_pinning.py).
    eta = max(homog.grid_spacing) if args.eta is None else args.eta
    Reg = (
        NodalPhaseFieldRegularization
        if args.density == "nodal"
        else PhaseFieldRegularization
    )
    with timer("setup"):
        reg = Reg(homog, eta=eta, weight=args.reg_weight)
        problem = StressTargetProblem(homog, cases, regularization=reg,
                                      design=args.density,
                                      hessian=(optimizer == "tr"),
                                      hv_warm_start=args.hv_warm_start)

    if args.init in INITIAL_DENSITY_KINDS:
        length = args.init_length
        if args.init == "filtered_random" and length is None:
            length = 3.0 * reg.eta
        with timer("initial density"):
            rho0 = initial_density(
                tuple(args.nb_grid_pts),
                kind=args.init,
                volume_fraction=args.init_volume_fraction,
                seed=args.seed,
                length=length,
                grid_spacing=homog.grid_spacing,
                subdomain_locations=homog.engine.subdomain_locations,
                nb_subdomain_grid_pts=homog.nb_pixels,
            )
    else:
        # Restart from the last frame of a previous run (every rank reads the
        # global field through muGrid; Fourier-resampled if the grids differ),
        # then slice out this rank's subdomain.
        with timer("initial density"):
            rho0_global, restart_meta = restart_density(
                args.init, args.nb_grid_pts)
        if rank0:
            src_grid = restart_meta["nb_grid_pts"]
            resample_str = (
                f", Fourier-resampled {tuple(src_grid)} -> "
                f"{tuple(args.nb_grid_pts)}"
                if restart_meta["resampled"] else "")
            print(f"init: last frame of {args.init}{resample_str}")
            if (restart_meta["domain_lengths"] is not None
                    and not np.allclose(restart_meta["domain_lengths"],
                                        homog.domain_lengths)):
                print(
                    f"WARNING: restart file has domain lengths "
                    f"{restart_meta['domain_lengths']}, this run uses "
                    f"{homog.domain_lengths}; the design is stretched onto "
                    "the new cell")
        rho0 = rho0_global[
            tuple(slice(lo, lo + n) for lo, n in
                  zip(homog.engine.subdomain_locations, homog.nb_pixels))
        ].copy()

    if rank0:
        print(
            f"muTopOpt: {dim}D  grid={tuple(args.nb_grid_pts)}  "
            f"load cases={len(cases)}  preconditioner={args.preconditioner}  "
            f"device={args.device}  precision={args.precision}"
        )
        if cg_tol_start is not None and cg_tol_start > cg_tol_min:
            cg_info = (f"adaptive cg-rtol {cg_tol_start:.0e} -> "
                       f"{cg_tol_min:.0e}")
        else:
            fixed = cg_tol_start if cg_tol_start is not None else args.cg_tol
            cg_info = f"fixed cg-rtol {fixed:.0e}"
        print(f"optimizer: {optimizer}"
              + ("  (auto)" if args.optimizer == "auto" else "")
              + f"  {cg_info}  gtol {bfgs_gtol:.3g}")
        feas = target_feasibility(dim, target_K, target_G,
                                  args.solid_E, args.solid_nu)
        if np.isfinite(feas["phi_min"]):
            print(f"target: K={target_K:.4g}  G={target_G:.4g}  "
                  f"E={target_E:.4g}  nu={target_nu:.4g}  ->  "
                  f"min solid fraction {feas['phi_min']:.2f} "
                  f"(Hashin-Shtrikman: K {feas['phi_hs_K']:.2f}, "
                  f"G {feas['phi_hs_G']:.2f}; Voigt: {feas['phi_voigt']:.2f})")
        for msg in feas["warnings"]:
            print(f"WARNING: {msg}")

    # Per-iteration L-BFGS history, collected across ranks with global
    # reductions so every rank holds the same series (safe to write below).
    n_global = comm.sum(float(rho0.size))
    hist = {"objective": [], "volume_fraction": [], "cg_iters": [],
            "hv_cg_iters": [], **{m: [] for m in _MODULI}}
    # Homogenized stresses of the most recent *accepted* iterate. problem.last
    # may instead hold a rejected trust-region trial or a line-search probe, so
    # the final frame and summary use this.
    accepted_stresses = [None]

    # `output.dump_every N` (N>0) streams the initial density (iteration 0) and every
    # N-th iterate to the output file as successive frames; the final iterate is
    # always added. Frames are written *directly* as they are produced (never
    # buffered in memory, which would overflow for large 3D grids), so the
    # NetCDF file is opened before the optimizer runs. muGrid freezes the header
    # on the first frame, so all global attributes must be declared now; the
    # ones only known at the end (final state, per-iteration histories,
    # frame_iterations) are declared at their maximum size with placeholders.
    # They are refreshed after every dumped frame (padded back to that maximum
    # size), so a run killed before it finishes, e.g. at the walltime, still
    # leaves a consistent record up to its last frame. The run shrinks them to
    # their true length only at the end: update_global_attribute() may shrink
    # an attribute but never grow it again.
    dump_every = args.dump_every
    dump_intermediate = (
        args.output is not None and dump_every is not None and dump_every > 0
    )
    fio = None
    field = None
    fdg_view = None  # numpy view of the per-frame applied-deformation-gradient
    stress_view = None  # numpy view of the per-frame homogenized stresses
    moduli_views = {}  # numpy views of the per-frame effective moduli
    frame_fields = ["density"]  # fields/variables written on every frame
    frame_iters = []  # L-BFGS iteration index of each frame actually written
    _MSG_LEN = 256  # fixed reservation for the optimizer message string
    if args.output is not None:
        field = homog.scalar_field("density")
        fio = muGrid.FileIONetCDF(
            args.output, muGrid.FileIONetCDF.OpenMode.Overwrite, comm
        )
        # Only the density is written; the solver scratch fields (`to_prob_*`)
        # in the same collection are deliberately excluded.
        fio.register_field_collection(homog.fc, field_names=["density"])
        # The applied deformation gradient is a per-frame, grid-less quantity
        # (one tensor per load case for the whole cell), so it is stored as a
        # frame variable rather than a field. It is constant here, so its buffer
        # is filled once and written on every frame.
        fdg_view = fio.register_frame_variable(
            "applied_deformation_gradient",
            list(applied_deformation_gradient.shape),
            np.float64,
        )
        fdg_view[...] = applied_deformation_gradient
        frame_fields.append("applied_deformation_gradient")
        # The homogenized stress of every load case for the frame's density,
        # i.e. the response to applied_deformation_gradient (enough to rebuild
        # the effective stiffness), and the isotropic-equivalent moduli derived
        # from it, in the same convention as the target_* attributes below.
        # They are NaN for frame 0: the initial density is stored before the
        # optimizer has evaluated it.
        stress_view = fio.register_frame_variable(
            "homogenized_stress", [len(cases), dim, dim], np.float64)
        frame_fields.append("homogenized_stress")
        for m in _MODULI:
            moduli_views[m] = fio.register_frame_variable(
                f"effective_{m}", [], np.float64)
            frame_fields.append(f"effective_{m}")
        # Target moduli. Both parameterizations are stored, whichever was
        # given on the command line.
        for m in _MODULI:
            fio.write_global_attribute(f"target_{m}", [float(targets[m])])
        # Physical cell size (per-file constant): a global attribute is correct.
        fio.write_global_attribute(
            "domain_lengths", [float(x) for x in homog.domain_lengths]
        )
        # Grid shape and scalar precision of the stored density. Both are
        # needed to *read* the file back (muGrid must register a field of the
        # stored shape and dtype before it can read, and its Python API cannot
        # inquire them from the file), so restarts from such a file rely on
        # these; see muTopOpt.restart.
        fio.write_global_attribute(
            "nb_grid_pts", [int(n) for n in args.nb_grid_pts]
        )
        fio.write_global_attribute("precision", args.precision)
        # "managed", "plain" or "host": it changes the speed of a multi-GPU
        # run several-fold, so it belongs next to the other run parameters.
        fio.write_global_attribute("device_memory", homog.device_memory)
        # Whether muGrid hands device buffers to MPI or stages them through
        # the host: the other half of a multi-GPU run's speed.
        fio.write_global_attribute("gpu_aware_mpi",
                                   "yes" if gpu_aware_mpi() else "no")
        # Full invocation that produced this file, quoted so it can be pasted
        # back into a shell to reproduce the run. Known up front, so written
        # with its final value here (no placeholder/update). `sys.argv` is
        # identical across ranks, so writing from all ranks is consistent.
        fio.write_global_attribute("command_line", shlex.join(sys.argv))
        # The complete control (file values plus defaults), so the file alone
        # reproduces the run: save this attribute as a .json and pass it to
        # mutopopt.
        fio.write_global_attribute("control", dumps(control))
        # Which build produced the file. `command_line` says what was asked
        # for; these say what ran it, which is what distinguishes two otherwise
        # identical runs whose numbers differ.
        for name, value in provenance_attributes().items():
            fio.write_global_attribute(name, value)
        # Placeholders, sized to the maximum they can reach (the optimizer runs
        # at most args.bfgs_maxiter iterations, hence at most that many history
        # entries and dumped frames). Entries not yet reached are NaN (float)
        # or -1 (integer) while the run is in progress.
        maxlen = int(args.bfgs_maxiter) + 1
        max_frames = (maxlen // dump_every + 3) if dump_intermediate else 1
        nan = float("nan")
        fio.write_global_attribute("dump_every", [int(dump_every)])
        fio.write_global_attribute("converged", [0])
        fio.write_global_attribute("optimizer_message", " " * _MSG_LEN)
        # Number of *accepted* iterates -- the index the per-iteration
        # histories and `frame_iterations` below share.
        fio.write_global_attribute("nb_iterations", [0])
        # The optimizer's raw step count. For the trust region this also
        # counts rejected trial steps, so it exceeds `nb_iterations`; the gap
        # is work that produced no iterate. Only known once the optimizer
        # returns, so -1 while the run is in progress.
        fio.write_global_attribute("nb_optimizer_steps", [-1])
        fio.write_global_attribute("final_objective", [nan])
        fio.write_global_attribute("final_max_gradient", [nan])
        fio.write_global_attribute("lbfgs_objective_history", [nan] * maxlen)
        fio.write_global_attribute("lbfgs_volume_fraction_history", [nan] * maxlen)
        fio.write_global_attribute("lbfgs_cg_iters_history", [-1] * maxlen)
        # Per-iteration Hessian-vector-product CG iterations (trust region;
        # all zero for L-BFGS) -- the dominant, otherwise-unrecorded cost.
        fio.write_global_attribute("hv_cg_iters_history", [-1] * maxlen)
        # Per-iteration effective moduli of the accepted iterate (compare to
        # the target_* attributes).
        for m in _MODULI:
            fio.write_global_attribute(f"effective_{m}_history", [nan] * maxlen)
        fio.write_global_attribute("frame_iterations", [-1] * max_frames)

    # Flush each frame to disk as it is written, so the output can be
    # inspected while the (possibly long) optimization is still running. The
    # PnetCDF backend commits collective writes on its own; the classic /
    # NetCDF-4 backend otherwise buffers frames until close(). sync() needs a
    # recent enough muGrid (feature-detected on the instance); disabled with
    # output.flush_frames.
    flush_frames = (not args.no_flush) and hasattr(fio, "sync")

    def write_frame(it, rho, stresses=None):
        """Stream one density iterate, the applied deformation gradient and
        the iterate's homogenized stresses and effective moduli (NaN if
        `stresses` is None) to the output as a new frame."""
        with timer("file output"):
            field.p[...] = homog.to_device(rho)
            if stresses is None:
                stress_view[...] = np.nan
                for m in _MODULI:
                    moduli_views[m][...] = np.nan
            else:
                stress_view[...] = np.stack(stresses)
                for m, value in moduli_of(stresses).items():
                    moduli_views[m][...] = value
            fio.append_frame().write(frame_fields)
            if flush_frames:
                fio.sync()
            frame_iters.append(int(it))

    def write_run_state(info=None):
        """Write the run-state attributes declared as placeholders above.

        With `info=None` this is a checkpoint of a run still in progress: the
        values reflect the last accepted iterate, and the per-iteration arrays
        are padded back to their reserved length so later checkpoints can
        still grow them. With the optimizer's final `info` the attributes get
        their final values and their true length. Every value is globally
        consistent across ranks (the histories are built from global
        reductions, and NuMPI's optimizers return the same result on every
        rank), so updating from all ranks is safe."""
        final = info is not None

        def upd(name, value):
            fio.update_global_attribute(name, name, value)

        def series(values, length, fill):
            values = list(values)
            return values if final else values + [fill] * (length - len(values))

        nb_accepted = len(hist["objective"])
        if final:
            upd("converged", [int(bool(info["success"]))])
            upd("optimizer_message", str(info["message"])[:_MSG_LEN])
            upd("nb_optimizer_steps", [int(info["nit"])])
            upd("final_objective", [float(info["objective"])])
            upd("final_max_gradient", [float(info["max_grad"])])
        else:
            # Same length as the reservation, so the final message can still
            # be written.
            upd("optimizer_message",
                f"RUNNING (checkpoint at iteration {nb_accepted})"
                .ljust(_MSG_LEN)[:_MSG_LEN])
            upd("final_objective",
                [hist["objective"][-1] if hist["objective"] else nan])
            controller = getattr(problem, "inner_tolerance", None)
            gnorm = (controller.latest_gnorm if controller is not None
                     else None)
            upd("final_max_gradient",
                [float(gnorm) if gnorm is not None else nan])
        upd("nb_iterations", [nb_accepted])
        # On the final write an empty history keeps its placeholder: an
        # attribute cannot be shrunk to zero length.
        if nb_accepted or not final:
            upd("lbfgs_objective_history",
                series(hist["objective"], maxlen, nan))
            upd("lbfgs_volume_fraction_history",
                series(hist["volume_fraction"], maxlen, nan))
            upd("lbfgs_cg_iters_history", series(hist["cg_iters"], maxlen, -1))
            upd("hv_cg_iters_history", series(hist["hv_cg_iters"], maxlen, -1))
            for m in _MODULI:
                upd(f"effective_{m}_history", series(hist[m], maxlen, nan))
        upd("frame_iterations", series(frame_iters, max_frames, -1))

    # Initial configuration as frame 0 (only when dumping intermediate steps).
    if dump_intermediate:
        write_frame(0, rho0)

    # Inner solves cut short by the CG stagnation safeguard (see
    # Homogenization.solve_rhs): `cg_stagnation_count` accumulates over the
    # run; the callback reports the per-outer-iterate delta.
    stall_seen = [0]

    # Wall-clock reference of the per-iteration timing printed with each outer
    # iterate; reset right before the optimizer starts.
    t_iter = [time.perf_counter()]

    def cb(it, rho, last):
        # Wall time of this outer iterate: everything since the previous
        # callback (line search / trust-region solves, frame dump, ...).
        now = time.perf_counter()
        dt, t_iter[0] = now - t_iter[0], now
        vf = comm.sum(float(np.sum(rho))) / n_global
        cg = last.get("cg_iters", [])
        cg_total = int(sum(cg))
        stalled = homog.cg_stagnation_count - stall_seen[0]
        stall_seen[0] = homog.cg_stagnation_count
        # Trust-region Hessian-vector-product CG work for this outer step
        # (0 for L-BFGS); the state/adjoint `cg_total` above alone badly
        # understates the trust-region cost.
        hv_cg = int(last.get("hv_cg_iters", 0))
        nb_hessp = int(last.get("nb_hessp", 0))
        hist["objective"].append(float(last["objective"]))
        hist["volume_fraction"].append(vf)
        hist["cg_iters"].append(cg_total)
        hist["hv_cg_iters"].append(hv_cg)
        accepted_stresses[0] = [np.array(s, copy=True)
                                for s in last["stresses"]]
        moduli = moduli_of(accepted_stresses[0])
        for m in _MODULI:
            hist[m].append(moduli[m])
        if dump_intermediate and it % dump_every == 0:
            write_frame(it, rho, accepted_stresses[0])
            with timer("file output"):
                write_run_state()
                if flush_frames:
                    fio.sync()
        if rank0:
            # Per-CG-iteration residuals are reported live during the solves
            # themselves (inner_solver.print_iterations); here we always summarize the
            # outer step, the current effective moduli against their targets,
            # and the inner CG iterations it took (the same totals also go to
            # the NetCDF history above).
            K, G, E, nu = (moduli[m] for m in _MODULI)
            rtol = last.get("cg_rtol")
            rtol_str = f"  cg-rtol={rtol:.1e}" if rtol is not None else ""
            # Solves that hit their finite-precision floor and returned a
            # best-effort iterate this outer step (0 = all solves converged).
            rtol_str += f"  cg-stalled={stalled}" if stalled else ""
            hv_str = (f"  hv-cg={hv_cg} ({nb_hessp} Hv)"
                      if nb_hessp else "")
            iter_label = "tr-iter" if optimizer == "tr" else "bfgs-iter"
            print(
                f"  {iter_label} {it:4d}  f={last['objective']:.6e}  "
                f"vol_frac={vf:.3f}  K={K:.4g} (target {target_K:.4g})  "
                f"G={G:.4g} (target {target_G:.4g})  "
                f"E={E:.4g} (target {target_E:.4g})  "
                f"nu={nu:.4g} (target {target_nu:.4g})  cg-iters={cg_total}"
                f"{hv_str}{rtol_str}  t={dt:.2f}s"
            )

    t_iter[0] = time.perf_counter()
    with timer("optimization"):
        if optimizer == "tr":
            rho, info = optimize_trust_region(
                problem,
                rho0,
                comm=comm,
                maxiter=args.bfgs_maxiter,
                gtol=bfgs_gtol,
                delta0=args.tr_delta0,
                delta_max=args.tr_delta_max,
                eta=args.tr_eta,
                cg_tol_start=cg_tol_start,
                cg_tol_min=cg_tol_min,
                hv_rtol=args.hv_cg_tol,
                callback=cb,
            )
        else:
            rho, info = optimize_bounded_lbfgs(
                problem,
                rho0,
                comm=comm,
                maxiter=args.bfgs_maxiter,
                gtol=bfgs_gtol,
                xtol=args.bfgs_xtol,
                callback=cb,
                cg_tol_start=cg_tol_start,
                cg_tol_min=cg_tol_min,
                cg_forcing_exp=args.cg_forcing_exp,
                cg_stall_rel=args.cg_stall_rel,
                cg_stall_shrink=args.cg_stall_shrink,
            )
    if optimizer == "tr" and rank0:
        print(
            f"trust region: {info['nb_hessp']} Hessian products, "
            f"final state-solve rtol {info['final_cg_rtol']:.1e}"
        )

    converged = bool(info["success"])
    # Two distinct counters exist and must not be conflated. `nb_accepted` is
    # the number of accepted iterates -- what the per-iteration lines above,
    # the histories and the dumped frames are all indexed by. `info["nit"]` is
    # the optimizer's raw step count, which for the trust region also includes
    # rejected trial steps. Report the accepted count, so the line below agrees
    # with the last per-iteration line, and name the raw count separately when
    # the two differ.
    nb_accepted = len(hist["objective"])
    nb_steps = int(info["nit"])
    final_stresses = (accepted_stresses[0] if accepted_stresses[0] is not None
                      else problem.last["stresses"])
    if rank0:
        K, G = effective_moduli(final_stresses)
        E, nu = E_nu_from_K_G(K, G)
        opt_label = "trust-region" if optimizer == "tr" else "optimizer"
        steps_str = ("" if nb_steps == nb_accepted else
                     f" ({nb_steps} {opt_label} steps incl. rejected)")
        print(
            f"done: {info['message']}  f={info['objective']:.6e}  "
            f"iters={nb_accepted}{steps_str}  "
            f"K={K:.4g} (target {target_K:.4g})  "
            f"G={G:.4g} (target {target_G:.4g})  "
            f"E={E:.4g} (target {target_E:.4g})  "
            f"nu={nu:.4g} (target {target_nu:.4g})"
        )
        if not converged:
            opt_name = "trust-region" if optimizer == "tr" else "L-BFGS"
            print(
                f"WARNING: {opt_name} did NOT converge; the written density "
                "is the last (non-converged) iterate (converged=0 in the "
                "output file)."
            )

    if args.output is not None:
        # Always include the final iterate as the last frame (unless it was
        # already the most recent one dumped, i.e. its iteration is a multiple
        # of output.dump_every).
        # The final frame is labelled with the accepted-iterate index, the
        # same counter `write_frame` uses for the intermediate dumps: then
        # `frame_iterations` holds a single counter, and the duplicate check
        # below recognizes an iterate that was already dumped (the raw step
        # count info["nit"] also counts rejected trust-region steps).
        final_it = nb_accepted
        if not frame_iters or frame_iters[-1] != final_it:
            write_frame(final_it, rho, final_stresses)

        # Give the placeholders declared before the frames their final values
        # and true length. Each update is same-or-smaller than its placeholder,
        # which muGrid permits (the frozen header cannot grow).
        #
        # `converged` is the machine-readable flag (1 = the optimizer met its
        # tolerances, 0 = it stopped early, e.g. at maxiter); the remaining
        # attributes give the reason, the final optimizer state, and the
        # per-iteration histories (one value per outer iteration, for later
        # plotting). `frame_iterations` records the iteration each frame holds
        # (0 = initial configuration), so a reader can map frames to iterates.
        with timer("file output"):
            write_run_state(info)
            fio.close()
        if rank0:
            print(
                f"wrote {args.output} ({len(frame_iters)} frame(s), "
                f"converged={int(converged)})"
            )

    # Full hierarchical timing breakdown. Collective when parallel: every rank
    # participates in the gather, rank 0 prints a single min/mean/max summary.
    timer.print_summary(title="muTopOpt timing summary", comm=comm)


if __name__ == "__main__":
    main()
