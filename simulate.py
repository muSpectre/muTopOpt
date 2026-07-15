#!/usr/bin/env python3
#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Command-line driver for muTopOpt: optimize a density unit cell for a target
isotropic effective stiffness, in 2D or 3D. The target is given either as bulk
and shear moduli (``--target-K``/``--target-G``) or as Young's modulus and
Poisson's ratio (``--target-E``/``--target-nu``).

By default the density is a *nodal* finite-element field with the fully
consistent Galerkin phase-field regularization, started from a filtered random
field -- the combination least prone to locking in the initial topology.

Examples
--------
    python simulate.py -n 64 64            --target-K 0.1 --target-G 0.05
    python simulate.py -n 64 64            --target-E 0.2 --target-nu -0.3
    python simulate.py -n 96 96 96 --bfgs-maxiter 300 --eta 0.02
    mpirun -np 4 python simulate.py -n 128 128 128     # (serial optimizer; see notes)

The solve/sensitivity are FFT-accelerated, J-FFT-preconditioned and (with a GPU
build of muGrid + ``--device gpu``) run on device. The outer L-BFGS optimizer is
currently serial.
"""

import argparse
import os
import shlex
import sys

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


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Keep the raw docstring layout *and* append each option's default."""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=_HelpFormatter)
    p.add_argument(
        "-n",
        "--nb-grid-pts",
        type=int,
        nargs="+",
        required=True,
        help="grid points per axis (2 or 3 values)",
    )
    p.add_argument(
        "--domain-lengths",
        type=float,
        nargs="+",
        default=None,
        help="physical edge length of the unit cell per axis (2 or 3 values, "
        "matching -n); default: unit length on every axis",
    )
    p.add_argument("--solid-E", type=float, default=1.0, help="solid Young's modulus")
    p.add_argument("--solid-nu", type=float, default=0.3, help="solid Poisson's ratio")
    p.add_argument("--penalty", type=float, default=2.0, help="SIMP exponent p")
    p.add_argument(
        "--void-ratio", type=float, default=0.0, help="void/solid stiffness ratio"
    )
    p.add_argument("--target-K", type=float, default=0.1, help="target bulk modulus")
    p.add_argument("--target-G", type=float, default=0.05, help="target shear modulus")
    p.add_argument(
        "--target-E",
        type=float,
        default=None,
        help="target Young's modulus (alternative to --target-K/--target-G; "
        "requires --target-nu)",
    )
    p.add_argument(
        "--target-nu",
        type=float,
        default=None,
        help="target Poisson's ratio (alternative to --target-K/--target-G; "
        "requires --target-E)",
    )
    p.add_argument(
        "--eta",
        type=float,
        default=None,
        help="phase-field interface width, in physical length units "
        "(default: two grid spacings)",
    )
    p.add_argument(
        "--reg-weight",
        type=float,
        default=1.0,
        help="overall strength of the phase-field regularization",
    )
    p.add_argument(
        "--init-volume-fraction",
        type=float,
        default=0.5,
        help="volume fraction of the initial density field",
    )
    p.add_argument(
        "--init",
        default="filtered_random",
        metavar="KIND_OR_FILE",
        help="initial density field: 'uniform' (constant), 'random' "
        "(white noise), 'filtered_random' (noise smoothed to a correlation "
        "length; least prone to locking the initial topology), or the NetCDF "
        "output of a previous run to restart from its last frame (Fourier-"
        "resampled if the stored grid does not match -n)",
    )
    p.add_argument(
        "--init-length",
        type=float,
        default=None,
        help="correlation length for --init filtered_random (default: 3*eta)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for the --init random/filtered_random density field",
    )
    p.add_argument(
        "--optimizer",
        choices=["auto", "lbfgs", "tr"],
        default="auto",
        help="outer optimizer: 'tr' (trust-region Newton-CG with exact "
        "second-order-adjoint Hessian-vector products; robust against "
        "loose inner CG tolerances, at ~2 extra solves per load case "
        "per Hessian product; needs NuMPI with tr_newton_bounded), "
        "'lbfgs' (bound-constrained L-BFGS), or 'auto' (tr when the "
        "installed NuMPI supports it, else lbfgs)",
    )
    p.add_argument(
        "--bfgs-maxiter",
        type=int,
        default=200,
        help="maximum number of outer optimizer iterations (L-BFGS or TR)",
    )
    p.add_argument(
        "--tr-delta0",
        type=float,
        default=0.05,
        help="initial trust-region radius, in RMS density change per pixel",
    )
    p.add_argument(
        "--tr-delta-max",
        type=float,
        default=0.5,
        help="maximum trust-region radius, in RMS density change per pixel",
    )
    p.add_argument(
        "--tr-eta",
        type=float,
        default=0.1,
        help="trust-region step acceptance threshold on ared/pred",
    )
    p.add_argument(
        "--hv-cg-tol",
        type=float,
        default=1e-3,
        help="relative CG tolerance of the two extra solves per "
        "Hessian-vector product (may be loose: the Hessian only shapes "
        "the trust-region model)",
    )
    p.add_argument(
        "--output-cg-iters",
        action="store_true",
        help="print one line per inner CG iteration (residual and "
        "relative residual) for every forward/adjoint solve, "
        "so the CG convergence can be watched live",
    )
    p.add_argument(
        "--bfgs-gtol",
        type=float,
        default=None,
        help="convergence tolerance on the projected gradient (L-BFGS and "
        "TR), measured on the mesh-invariant volume-fraction derivative "
        "(V/V_pixel)*df/drho -- the same value means the same physical "
        "stationarity at every resolution (typical initial designs start "
        "at ~100 in these units). Default: 2.5 in double precision, 25 in "
        "single precision (the float32 solve accuracy floor makes a "
        "tighter gradient tolerance uncertifiable)",
    )
    p.add_argument(
        "--bfgs-xtol",
        type=float,
        default=0.0,
        help="L-BFGS convergence tolerance on the step size (relative change "
        "in the density iterate); 0 disables the criterion",
    )
    p.add_argument(
        "--cg-tol",
        type=float,
        default=1e-4,
        help="inner CG relative tolerance; loose values are safe "
        "because the objective is adjoint-corrected "
        "(Lagrangian) and thus second-order accurate in the "
        "solve error",
    )
    p.add_argument(
        "--cg-maxiter",
        type=int,
        default=2000,
        help="maximum number of inner CG iterations per solve "
        "(the solve stops here even if --cg-tol is not met)",
    )
    p.add_argument(
        "--cg-tol-start",
        type=float,
        default=1e-2,
        help="adaptive inner CG tolerance (ON by default): solves start at "
        "this (coarse) relative tolerance and tighten automatically -- "
        "for the trust-region optimizer driven by the computable "
        "|f_err| <= eta_f*pred accuracy condition, for L-BFGS by an "
        "Eisenstat-Walker forcing term with a stagnation ratchet. Pass "
        "0 (or negative) to disable adaptation and use the fixed "
        "--cg-tol throughout",
    )
    p.add_argument(
        "--cg-tol-min",
        type=float,
        default=None,
        help="floor for the adaptive inner tolerance. Default for L-BFGS: "
        "--bfgs-gtol/1e4, capped at --cg-tol (the mesh-invariant "
        "gradient error stays below ~1e3 x the CG tolerance -- measured "
        "at 32^3/64^3 with a 10x margin -- so this floor lets L-BFGS "
        "certify convergence at --bfgs-gtol); 1e-10 for the trust-region "
        "optimizer (its accuracy control only tightens as far as the "
        "predicted reduction requires). In single precision the "
        "default floor is clamped to 1e-6: below that the true "
        "residual b-Kx of a float32 solve stagnates (only the "
        "recursive CG residual keeps shrinking) and CG may fail "
        "outright",
    )
    p.add_argument(
        "--cg-forcing-exp",
        type=float,
        default=1.0,
        help="exponent alpha in the relative forcing term "
        "rtol = rtol_start * (||g_free||/||g_0||)**alpha (default 1.0; "
        "alpha=1 gives rtol=O(||g||) and fast local convergence)",
    )
    p.add_argument(
        "--cg-stall-shrink",
        type=float,
        default=0.3,
        help="stagnation ratchet for the adaptive tolerance: when the "
        "projected gradient stops decreasing, the inner tolerance is "
        "multiplied by this factor (default 0.3) toward --cg-tol-min, so a "
        "noise-limited plateau cannot stall the run",
    )
    p.add_argument(
        "--cg-stall-rel",
        type=float,
        default=1e-2,
        help="minimum relative decrease in the projected gradient counted as "
        "progress (default 0.01); an iterate that does not beat the best "
        "gradient by this much triggers the --cg-stall-shrink ratchet",
    )
    p.add_argument(
        "--preconditioner",
        choices=["green-jacobi", "green"],
        default="green-jacobi",
        help="inner-solve preconditioner: 'green-jacobi' (J-FFT, "
        "reference stiffness times a per-pixel Jacobi scaling) "
        "or 'green' (plain reference-stiffness Green operator)",
    )
    p.add_argument(
        "--element",
        choices=["p1", "q1"],
        default="q1",
        help="finite element (P1 simplices or Q1 hex/quad)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="run the forward/adjoint solves and sensitivity on the "
        "host ('cpu') or on the accelerator ('gpu', or the platform-"
        "specific 'rocm'/'cuda'); the L-BFGS optimizer always runs on "
        "the host. Append ':N' to pin a specific device by id (e.g. "
        "'rocm:0', 'cuda:1') instead of the default per-rank placement",
    )
    p.add_argument(
        "--precision",
        choices=["single", "double"],
        default="double",
        help="scalar precision of the on-grid fields and the "
        "FFT-accelerated solves (the FFT engine dispatches on the "
        "field dtype); the host L-BFGS optimizer stays double",
    )
    p.add_argument(
        "--density",
        choices=["element", "nodal"],
        default="nodal",
        help="density discretization: 'nodal' (nodal FE field, "
        "SIMP on the element average of the interpolant, fully "
        "consistent Galerkin phase-field regularization) or "
        "'element' (per-pixel density, FD Laplacian penalty)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="NetCDF file to write the optimized density to",
    )
    p.add_argument(
        "--dump-every",
        type=int,
        default=-1,
        help="dump intermediate L-BFGS iterates to the NetCDF output "
        "as successive frames: with N>0 the initial "
        "configuration and every N-th iterate (N, 2N, ...) are "
        "written, and the final iterate is always included; -1 "
        "(default) writes only the final density as a single "
        "frame",
    )
    p.add_argument(
        "--no-flush",
        action="store_true",
        help="do not flush each frame to disk as it is written. By default "
        "every frame is synced (muGrid FileIONetCDF.sync) so the output can "
        "be inspected mid-run; the classic/NetCDF-4 backend otherwise "
        "buffers frames until the file is closed (PnetCDF commits on its "
        "own). Needs a muGrid that provides sync(); ignored otherwise",
    )
    args = p.parse_args()

    dim = len(args.nb_grid_pts)
    if dim not in (2, 3):
        p.error("-n takes 2 or 3 values")
    if args.domain_lengths is not None and len(args.domain_lengths) != dim:
        p.error(f"--domain-lengths must have {dim} values (one per axis)")
    if args.init not in INITIAL_DENSITY_KINDS and not os.path.exists(args.init):
        p.error(
            f"--init must be one of {', '.join(INITIAL_DENSITY_KINDS)} or an "
            f"existing NetCDF restart file; '{args.init}' is neither")

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
    # derivative ĝ = (V/V_pixel) df/drho (see muTopOpt.optimize.
    # _gradient_scale) -- the raw per-pixel gradient shrinks like 1/N with
    # resolution, so an absolute per-pixel gtol falsely certified untouched
    # initial designs as converged on fine grids. Typical initial designs
    # start at ||ĝ||_inf ~ O(100); the defaults reproduce the pre-scaling
    # 64^3 behavior (1e-4 per-pixel * 64^3 ~ 26) at every resolution. The
    # single-precision default stays 10x coarser: the achievable ĝ accuracy
    # at the float32 residual floor makes a tighter tolerance uncertifiable.
    single = args.precision == "single"
    rtol_floor = 1e-6 if single else 0.0
    bfgs_gtol = (args.bfgs_gtol if args.bfgs_gtol is not None
                 else (25.0 if single else 2.5))

    # Adaptive inner CG tolerance is on by default (<= 0 disables it: fixed
    # --cg-tol throughout). The floor defaults per optimizer: L-BFGS needs the
    # final gradient error below the gradient tolerance to certify
    # convergence -- in the mesh-invariant ĝ units the error is bounded by
    # ~1e3 x the CG tolerance (measured at 32^3/64^3 on the initial iterate;
    # _KAPPA_EFF below includes a 10x margin for late high-contrast designs),
    # so the floor is bfgs_gtol/_KAPPA_EFF, capped at the fixed --cg-tol. The
    # trust-region accuracy control only tightens as far as the predicted
    # reduction requires, so it just gets ample room.
    _KAPPA_EFF = 1e4  # mesh-invariant gradient error per unit CG rtol
    cg_tol_start = (args.cg_tol_start
                    if args.cg_tol_start and args.cg_tol_start > 0 else None)
    if args.cg_tol_min is not None:
        cg_tol_min = args.cg_tol_min
        if cg_tol_min < rtol_floor:
            import warnings

            warnings.warn(
                f"--cg-tol-min {cg_tol_min:.1e} is below the single-"
                f"precision solve accuracy floor (~{rtol_floor:.0e}); the "
                "true residual cannot reach it and CG may fail",
                RuntimeWarning,
            )
    elif optimizer == "tr":
        cg_tol_min = max(1e-10, rtol_floor)
    else:
        cg_tol_min = max(min(args.cg_tol, bfgs_gtol / _KAPPA_EFF), rtol_floor)
    if cg_tol_start is None and optimizer == "tr":
        # Adaptation disabled: pin the trust-region accuracy control at the
        # fixed --cg-tol (start == floor), i.e. a truly fixed tolerance.
        cg_tol_start = args.cg_tol
        cg_tol_min = args.cg_tol

    if muGrid.has_mpi:
        from mpi4py import MPI

        mpi_comm = MPI.COMM_WORLD
        comm = muGrid.Communicator(mpi_comm)
    else:
        mpi_comm = None
        comm = muGrid.Communicator()
    rank0 = comm.rank == 0

    material = SimpMaterial(args.solid_E, args.solid_nu, args.penalty, args.void_ratio)
    # CLI exposes "single"/"double"; muTopOpt works in NumPy dtypes internally.
    dtype = {"single": np.float32, "double": np.float64}[args.precision]
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
    if (args.target_E is None) != (args.target_nu is None):
        p.error("--target-E and --target-nu must be given together")
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
    # are the true 3D constants, matching --target-E/--target-nu).
    def E_nu_from_K_G(K, G):
        return E_nu_from_lame(K - 2.0 * G / dim, G)

    target_K, target_G = effective_moduli([lc.target_stress for lc in cases])
    target_E, target_nu = E_nu_from_K_G(target_K, target_G)
    # The interface width defaults to two grid spacings: wide enough that the
    # regularization can move interfaces (merge/remove features) instead of
    # freezing the initial topology, narrow enough for crisp designs.
    eta = max(homog.grid_spacing) if args.eta is None else args.eta
    Reg = (
        NodalPhaseFieldRegularization
        if args.density == "nodal"
        else PhaseFieldRegularization
    )
    reg = Reg(homog, eta=eta, weight=args.reg_weight)
    problem = StressTargetProblem(homog, cases, regularization=reg,
                                  design=args.density,
                                  hessian=(optimizer == "tr"))

    if args.init in INITIAL_DENSITY_KINDS:
        length = args.init_length
        if args.init == "filtered_random" and length is None:
            length = 3.0 * reg.eta
        rho0 = initial_density(
            homog.nb_pixels,
            kind=args.init,
            volume_fraction=args.init_volume_fraction,
            seed=args.seed,
            length=length,
            grid_spacing=homog.grid_spacing,
        )
    else:
        # Restart from the last frame of a previous run (every rank reads the
        # global field through muGrid; Fourier-resampled if the grids differ),
        # then slice out this rank's subdomain.
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

    print(muGrid.version_string(communicator=homog.comm, device=homog.device))
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
            "hv_cg_iters": []}

    # `--dump-every N` (N>0) streams the initial density (iteration 0) and every
    # N-th iterate to the output file as successive frames; the final iterate is
    # always added. Frames are written *directly* as they are produced (never
    # buffered in memory, which would overflow for large 3D grids), so the
    # NetCDF file is opened before the optimizer runs. muGrid freezes the header
    # on the first frame, so all global attributes must be declared now; the
    # ones only known at the end (final state, per-iteration histories,
    # frame_iterations) are declared at their maximum size with placeholders and
    # overwritten in place afterwards -- update_global_attribute() may shrink an
    # attribute but never grow it.
    dump_every = args.dump_every
    dump_intermediate = (
        args.output is not None and dump_every is not None and dump_every > 0
    )
    fio = None
    field = None
    fdg_view = None  # numpy view of the per-frame applied-deformation-gradient
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
        # Physical cell size (per-file constant): a global attribute is correct.
        fio.write_global_attribute(
            "domain_lengths", [float(x) for x in homog.domain_lengths]
        )
        # Grid shape and scalar precision of the stored density. Both are
        # needed to *read* the file back (muGrid must register a field of the
        # stored shape and dtype before it can read, and its Python API cannot
        # inquire them from the file), so `--init <file>` restarts rely on
        # these; see muTopOpt.restart.
        fio.write_global_attribute(
            "nb_grid_pts", [int(n) for n in args.nb_grid_pts]
        )
        fio.write_global_attribute("precision", args.precision)
        # Full invocation that produced this file, quoted so it can be pasted
        # back into a shell to reproduce the run. Known up front, so written
        # with its final value here (no placeholder/update). `sys.argv` is
        # identical across ranks, so writing from all ranks is consistent.
        fio.write_global_attribute("command_line", shlex.join(sys.argv))
        # Placeholders, sized to the maximum they can reach (the optimizer runs
        # at most args.bfgs_maxiter iterations, hence at most that many history
        # entries and dumped frames). Real values are written after the run.
        maxlen = int(args.bfgs_maxiter) + 1
        max_frames = (maxlen // dump_every + 3) if dump_intermediate else 1
        fio.write_global_attribute("dump_every", [int(dump_every)])
        fio.write_global_attribute("converged", [0])
        fio.write_global_attribute("optimizer_message", " " * _MSG_LEN)
        fio.write_global_attribute("nb_iterations", [0])
        fio.write_global_attribute("final_objective", [0.0])
        fio.write_global_attribute("final_max_gradient", [0.0])
        fio.write_global_attribute("lbfgs_objective_history", [0.0] * maxlen)
        fio.write_global_attribute("lbfgs_volume_fraction_history", [0.0] * maxlen)
        fio.write_global_attribute("lbfgs_cg_iters_history", [0] * maxlen)
        # Per-iteration Hessian-vector-product CG iterations (trust region;
        # all zero for L-BFGS) -- the dominant, otherwise-unrecorded cost.
        fio.write_global_attribute("hv_cg_iters_history", [0] * maxlen)
        fio.write_global_attribute("frame_iterations", [-1] * max_frames)

    # Flush each frame to disk as it is written, so the output can be
    # inspected while the (possibly long) optimization is still running. The
    # PnetCDF backend commits collective writes on its own; the classic /
    # NetCDF-4 backend otherwise buffers frames until close(). sync() needs a
    # recent enough muGrid (feature-detected on the instance); disabled with
    # --no-flush.
    flush_frames = (not args.no_flush) and hasattr(fio, "sync")

    def write_frame(it, rho):
        """Stream one density iterate (and the applied deformation gradient) to
        the output as a new frame."""
        field.p[...] = homog.to_device(rho)
        fio.append_frame().write(frame_fields)
        if flush_frames:
            fio.sync()
        frame_iters.append(int(it))

    # Initial configuration as frame 0 (only when dumping intermediate steps).
    if dump_intermediate:
        write_frame(0, rho0)

    # Inner solves cut short by the CG stagnation safeguard (see
    # Homogenization.solve_rhs): `cg_stagnation_count` accumulates over the
    # run; the callback reports the per-outer-iterate delta.
    stall_seen = [0]

    def cb(it, rho, last):
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
        if dump_intermediate and it % dump_every == 0:
            write_frame(it, rho)
        if rank0:
            # Per-CG-iteration residuals are reported live during the solves
            # themselves (--output-cg-iters); here we always summarize the
            # outer step, the current effective moduli against their targets,
            # and the inner CG iterations it took (the same totals also go to
            # the NetCDF history above).
            K, G = effective_moduli(last["stresses"])
            E, nu = E_nu_from_K_G(K, G)
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
                f"{hv_str}{rtol_str}"
            )

    if optimizer == "tr":
        rho, info = optimize_trust_region(
            problem,
            rho0,
            comm=mpi_comm,
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
        if rank0:
            print(
                f"trust region: {info['nb_hessp']} Hessian products, "
                f"final state-solve rtol {info['final_cg_rtol']:.1e}"
            )
    else:
        rho, info = optimize_bounded_lbfgs(
            problem,
            rho0,
            comm=mpi_comm,
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

    converged = bool(info["success"])
    if rank0:
        K, G = effective_moduli(problem.last["stresses"])
        E, nu = E_nu_from_K_G(K, G)
        print(
            f"done: {info['message']}  f={info['objective']:.6e}  "
            f"iters={info['nit']}  K={K:.4g} (target {target_K:.4g})  "
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
        # of --dump-every).
        final_it = int(info["nit"])
        if not frame_iters or frame_iters[-1] != final_it:
            write_frame(final_it, rho)

        # Overwrite the placeholders declared before the frames with the real
        # values now that the run has finished. Every value is globally
        # consistent across ranks (NuMPI's l_bfgs_bounded returns the same
        # result on every rank), so updating from all ranks is safe. Each update
        # is same-or-smaller than its placeholder, which muGrid permits (the
        # frozen header cannot grow).
        #
        # `converged` is the machine-readable flag (1 = the optimizer met its
        # tolerances, 0 = it stopped early, e.g. at maxiter); the remaining
        # attributes give the reason, the final optimizer state, and the
        # per-iteration L-BFGS histories (one value per outer iteration, for
        # later plotting). `frame_iterations` records the L-BFGS iteration each
        # frame holds (0 = initial configuration), so a reader can map frames to
        # iterates.
        def upd(name, value):
            fio.update_global_attribute(name, name, value)

        upd("converged", [int(converged)])
        upd("optimizer_message", str(info["message"])[:_MSG_LEN])
        upd("nb_iterations", [int(info["nit"])])
        upd("final_objective", [float(info["objective"])])
        upd("final_max_gradient", [float(info["max_grad"])])
        if hist["objective"]:
            upd("lbfgs_objective_history", hist["objective"])
            upd("lbfgs_volume_fraction_history", hist["volume_fraction"])
            upd("lbfgs_cg_iters_history", hist["cg_iters"])
            upd("hv_cg_iters_history", hist["hv_cg_iters"])
        upd("frame_iterations", frame_iters)
        fio.close()
        if rank0:
            print(
                f"wrote {args.output} ({len(frame_iters)} frame(s), "
                f"converged={int(converged)})"
            )


if __name__ == "__main__":
    main()
