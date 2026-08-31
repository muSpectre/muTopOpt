#!/usr/bin/env python3
#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Command-line driver for muTopOpt: optimize a density unit cell for a target
isotropic effective conductivity, in 2D or 3D.

By default the density is an *element-wise* finite-element field with the
phase-field regularization.

Examples
--------
    python simulate_conduction.py -n 64 64 --target-kappa 0.5
    python simulate_conduction.py -n 96 96 96 --target-kappa 0.3 --penalty 4.0
    mpirun -np 4 python simulate_conduction.py -n 128 128 128     # (serial optimizer; see notes)

The solve/sensitivity are FFT-accelerated, J-FFT-preconditioned and (with a GPU
build of muGrid + ``--device gpu``) run on device. The outer L-BFGS optimizer is
currently serial.
"""

import argparse
import os
import shlex
import sys
import time

import muGrid
import numpy as np

from muTopOpt import (
    FluxTargetProblem,
    HomogenizationConductivity,
    NodalPhaseFieldRegularization,
    PhaseFieldRegularization,
    SimpConductivity,
)
from muTopOpt.loadcases_conduction import target_load_cases
from muTopOpt.optimize import (
    initial_density,
    optimize_bounded_lbfgs,
)
from muTopOpt.restart import INITIAL_DENSITY_KINDS, restart_density


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Keep the raw docstring layout *and* append each option's default."""


def main():
    start = time.time()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=_HelpFormatter)
    p.add_argument(
        "-n",
        "--nb-grid-pts",
        type=int,
        nargs="+",
        # required=True,
        default=[32, 32],
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
    p.add_argument(
        "--kappa-solid",
        type=float,
        default=1.0,
        help="solid thermal conductivity",
    )
    p.add_argument(
        "--penalty",
        type=float,
        default=2.0,
        help="SIMP exponent p",
    )
    p.add_argument(
        "--void-ratio",
        type=float,
        default=1e-8,
        help="void/solid conductivity ratio",
    )
    p.add_argument(
        "--target-kappa",
        type=float,
        nargs="+",
        default=[0.4, 0, 0, 0.2],
        help="target conductivity: one value for an isotropic tensor "
             "(kappa * I) or dim*dim values for an arbitrary tensor "
             "(row-major, e.g. --target-kappa 0.5 0.1 0.1 0.5 for a 2x2 tensor)",
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
        default=1.,
        help="overall strength of the phase-field regularization",
    )
    p.add_argument(
        "--load-weight",
        type=float,
        default=1.,
        help="overall strength of the load cases",
    )
    p.add_argument(
        "--init-volume-fraction",
        type=float,
        default=0.9,
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
        "--bfgs-maxiter",
        type=int,
        default=200,
        help="maximum number of outer optimizer iterations (L-BFGS)",
    )
    p.add_argument(
        "--bfgs-gtol",
        type=float,
        default=None,
        help="convergence tolerance on the projected gradient (L-BFGS), "
             "measured on the mesh-invariant volume-fraction derivative "
             "(V/V_pixel)*df/drho -- the same value means the same physical "
             "stationarity at every resolution. Default: 2.5",
    )
    p.add_argument(
        "--bfgs-xtol",
        type=float,
        default=0.0,
        help="L-BFGS convergence tolerance on the step size (relative change "
             "in the density iterate); 0 disables the criterion",
    )

    p.add_argument(
        "--output-cg-iters",
        action="store_true",
        help="print one line per inner CG iteration (residual and "
             "relative residual) for every forward/adjoint solve",
    )
    p.add_argument(
        "--cg-tol",
        type=float,
        default=1e-4,
        help="inner CG relative tolerance",
    )
    p.add_argument(
        "--cg-maxiter",
        type=int,
        default=2000,
        help="maximum number of inner CG iterations per solve",
    )
    p.add_argument(
        "--cg-tol-start",
        type=float,
        default=1e-2,
        help="adaptive inner CG tolerance (ON by default): solves start at "
             "this (coarse) relative tolerance and tighten automatically via "
             "Eisenstat-Walker forcing term with a stagnation ratchet. Pass "
             "0 (or negative) to disable adaptation and use the fixed --cg-tol",
    )
    p.add_argument(
        "--cg-tol-min",
        type=float,
        default=None,
        help="floor for the adaptive inner tolerance. Default: "
             "--bfgs-gtol/1e4, capped at --cg-tol",
    )
    p.add_argument(
        "--cg-forcing-exp",
        type=float,
        default=1.0,
        help="exponent alpha in the relative forcing term",
    )
    p.add_argument(
        "--cg-stall-shrink",
        type=float,
        default=0.3,
        help="stagnation ratchet for the adaptive tolerance",
    )
    p.add_argument(
        "--cg-stall-rel",
        type=float,
        default=1e-2,
        help="minimum relative decrease in the projected gradient counted as "
             "progress",
    )
    p.add_argument(
        "--preconditioner",
        choices=["green-jacobi", "green"],
        default="green-jacobi",
        help="inner-solve preconditioner: 'green-jacobi' (J-FFT, "
             "Green operator times a per-node Jacobi diagonal assembled "
             "via a 2^dim-colour scheme) or 'green' (plain reference-"
             "conductivity Green operator)",
    )
    p.add_argument(
        "--element",
        choices=["p1", "q1"],
        default="p1",
        help="finite element (P1 simplices or Q1 hex/quad)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="run the forward/adjoint solves and sensitivity on the "
             "host ('cpu') or on the accelerator ('gpu')",
    )
    p.add_argument(
        "--precision",
        choices=["single", "double"],
        default="double",
        help="scalar precision of the on-grid fields and the FFT-accelerated solver",
    )
    p.add_argument(
        "--density",
        choices=["element", "nodal"],
        default="element",
        help="density discretization: 'nodal' or 'element'",
    )
    p.add_argument(
        "--output",
        type=str,
        default='test_output.nc',
        help="NetCDF file to write the optimized density to",
    )
    p.add_argument(
        "--temperature-rtol",
        type=float,
        default=1e-8,
        help="relative CG tolerance of the extra per-load-case solves that "
             "produce the stored temperature fields (one solve per load case "
             "per written frame)",
    )
    p.add_argument(
        "--dump-every",
        type=int,
        default=-1,
        help="dump intermediate L-BFGS iterates to the NetCDF output "
             "as successive frames: with N>0 the initial "
             "configuration and every N-th iterate (N, 2N, ...) are "
             "written, and the final iterate is always included; -1 "
             "(default) writes only the final density as a single frame",
    )
    p.add_argument(
        "--no-flush",
        action="store_true",
        help="do not flush each frame to disk as it is written",
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

    bfgs_gtol = args.bfgs_gtol if args.bfgs_gtol is not None else 2.5

    # Adaptive inner CG tolerance is on by default (<= 0 disables it).
    _KAPPA_EFF = 1e4
    cg_tol_start = (args.cg_tol_start
                    if args.cg_tol_start and args.cg_tol_start > 0 else None)
    if args.cg_tol_min is not None:
        cg_tol_min = args.cg_tol_min
    else:
        cg_tol_min = max(min(args.cg_tol, bfgs_gtol / _KAPPA_EFF), 0.0)

    if muGrid.has_mpi:
        from mpi4py import MPI

        mpi_comm = MPI.COMM_WORLD
        comm = muGrid.Communicator(mpi_comm)
    else:
        mpi_comm = None
        comm = muGrid.Communicator()
    rank0 = comm.rank == 0

    material = SimpConductivity(args.kappa_solid, args.penalty, args.void_ratio)
    dtype = {"single": np.float32, "double": np.float64}[args.precision]
    homog = HomogenizationConductivity(
        tuple(args.nb_grid_pts),
        material,
        comm=comm,
        element=args.element,
        domain_lengths=args.domain_lengths,
        preconditioner=args.preconditioner,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        cg_verbose=args.output_cg_iters,
        dtype=dtype,
    )

    # Target conductivity tensor: isotropic (scalar) or arbitrary (dim*dim values)
    if len(args.target_kappa) == 1:
        kappa_target = args.target_kappa[0] * np.eye(dim)
        kappa_isotropic = True
    elif len(args.target_kappa) == dim * dim:
        kappa_target = np.array(args.target_kappa, dtype=float).reshape(dim, dim)
        kappa_isotropic = False
    else:
        p.error(
            f"--target-kappa takes 1 value (isotropic) or {dim * dim} values "
            f"(full {dim}x{dim} tensor); got {len(args.target_kappa)}"
        )
    flux_magnitude = 1.0
    cases = target_load_cases(
        dim,
        lambda E: kappa_target @ E,
        weights=[args.load_weight] * dim,
    )

    # Effective conductivity tensor from the homogenized flux response.
    def effective_conductivity(fluxes):
        # fluxes is a list of flux responses, one per load case
        result = np.zeros((dim, dim))
        for i, (q, case) in enumerate(zip(fluxes, cases)):
            # Each load case is a unit gradient in direction i
            result[i] = q / flux_magnitude
        return result

    # The interface width defaults to two grid spacings.
    eta = max(homog.grid_spacing) if args.eta is None else args.eta
    Reg = (
        NodalPhaseFieldRegularization
        if args.density == "nodal"
        else PhaseFieldRegularization
    )
    reg = Reg(homog, eta=eta, weight=args.reg_weight)
    problem = FluxTargetProblem(homog, cases, regularization=reg)

    if args.init in INITIAL_DENSITY_KINDS:
        length = args.init_length
        if args.init == "filtered_random" and length is None:
            length = 3.0 * reg.eta
        # Generate the initial density on the full global grid so that MPI-parallel
        # runs start from exactly the same field as a serial run (the local
        # subdomain for each rank is just a slice of that global field).
        rho0_global = initial_density(
            tuple(args.nb_grid_pts),
            kind=args.init,
            volume_fraction=args.init_volume_fraction,
            seed=args.seed,
            length=length,
            grid_spacing=homog.grid_spacing,
        )
        rho0 = rho0_global[
            tuple(slice(lo, lo + n) for lo, n in
                  zip(homog.engine.subdomain_locations, homog.nb_pixels))
        ].copy()
    else:
        # Restart from a previous run
        rho0_global, restart_meta = restart_density(
            args.init, args.nb_grid_pts)
        if rank0:
            src_grid = restart_meta["nb_grid_pts"]
            resample_str = (
                f", Fourier-resampled {tuple(src_grid)} -> "
                f"{tuple(args.nb_grid_pts)}"
                if restart_meta["resampled"] else "")
            print(f"init: last frame of {args.init}{resample_str}")
        rho0 = rho0_global[
            tuple(slice(lo, lo + n) for lo, n in
                  zip(homog.engine.subdomain_locations, homog.nb_pixels))
        ].copy()

    print(muGrid.version_string(communicator=homog.comm))
    if rank0:
        print(
            f"muTopOpt (conductivity): {dim}D  grid={tuple(args.nb_grid_pts)}  "
            f"load cases={len(cases)}  preconditioner={args.preconditioner}  "
            f"device={args.device}  precision={args.precision}"
        )
        if cg_tol_start is not None and cg_tol_start > cg_tol_min:
            cg_info = (f"adaptive cg-rtol {cg_tol_start:.0e} -> "
                       f"{cg_tol_min:.0e}")
        else:
            fixed = cg_tol_start if cg_tol_start is not None else args.cg_tol
            cg_info = f"fixed cg-rtol {fixed:.0e}"
        print(f"optimizer: lbfgs  {cg_info}  gtol {bfgs_gtol:.3g}")
        if kappa_isotropic:
            print(f"target: kappa={args.target_kappa[0]:.4g} (isotropic)")
        else:
            print(f"target: kappa_tensor=\n{kappa_target} (anisotropic)")

    # Per-iteration L-BFGS history
    n_global = comm.sum(float(rho0.size))
    hist = {"objective": [], "volume_fraction": [], "cg_iters": []}

    dump_every = args.dump_every
    dump_intermediate = (
            args.output is not None and dump_every is not None and dump_every > 0
    )
    fio = None
    field = None
    # Temperature fields, one per unit macro-gradient load case. Created once
    # here and reused for every frame: muGrid writes whatever is in the
    # registered field collection, so the field object that is *solved into*
    # must be the same object that was registered.
    temperature_fields = {}
    # Total gradient and flux per load case, on the quadrature points. These
    # cannot reuse homog._g / homog._flux directly: those are single scratch
    # fields shared by all load cases and overwritten by the next solve.
    gradient_fields = {}
    flux_fields = {}
    # Every field written on each frame. muGrid opens a new frame per
    # append_frame() call, so all of these must go into a *single* write() --
    # one append_frame() per field would scatter density and temperatures
    # across separate frames, leaving each frame's other variables at their
    # NetCDF fill value (which is what "the temperatures are empty" looked
    # like on read-back).
    frame_fields = ["density"]
    frame_iters = []
    _MSG_LEN = 256

    def _like_flux(name):
        """A field with the same layout as homog._g / homog._flux: dim
        components on the quadrature points (2 per pixel for P1 in 2D)."""
        return homog.fc.real_field(
            name, (homog.dim,), "quad", dtype=homog.dtype)

    if args.output is not None:
        field = homog.scalar_field("density")
        fio = muGrid.FileIONetCDF(
            args.output, muGrid.FileIONetCDF.OpenMode.Overwrite, comm
        )
        fio.register_field_collection(homog.fc, field_names=["density"])

        for i in range(homog.dim):
            temperature_fields[i] = homog.scalar_field(f"temperature_{i}")
            gradient_fields[i] = _like_flux(f"gradient_{i}")
            flux_fields[i] = _like_flux(f"flux_{i}")
            for name in (f"temperature_{i}", f"gradient_{i}", f"flux_{i}"):
                fio.register_field_collection(homog.fc, field_names=[name])
                frame_fields.append(name)

        fio.write_global_attribute(
            "domain_lengths", [float(x) for x in homog.domain_lengths]
        )
        fio.write_global_attribute(
            "nb_grid_pts", [int(n) for n in args.nb_grid_pts]
        )
        fio.write_global_attribute("precision", args.precision)
        fio.write_global_attribute("command_line", shlex.join(sys.argv))
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
        fio.write_global_attribute("frame_iterations", [-1] * max_frames)

    flush_frames = (not args.no_flush) and hasattr(fio, "sync") if fio else False

    # Physical node positions x, for the affine part T_macro = g . x that
    # solve_macro does not include (it returns the periodic fluctuation only).
    # engine.coords is (dim, *nb_subdomain_grid_pts): integer grid indices in
    # some muFFT versions, normalized [0, 1) coordinates in others.
    _coords = np.asarray(homog.engine.coords)
    _scale = np.asarray(
        homog.grid_spacing if np.issubdtype(_coords.dtype, np.integer)
        else homog.domain_lengths, dtype=float)
    positions = _coords * _scale.reshape((dim,) + (1,) * dim)

    def write_frame(it, rho):
        """Stream one density iterate and its temperature fields to the output
        as a single new frame."""
        homog.set_density(rho)
        field.p[...] = homog.to_device(rho)

        # Populate every temperature field *before* the frame is opened, so
        # one write() can commit all of them together.
        macro_gradients = np.eye(homog.dim)
        for k in range(homog.dim):
            # Clear last frame's contents: the affine part added below is not
            # a valid starting iterate should solve_macro warm-start from the
            # field it is handed.
            temperature_fields[k].s[0] = 0.0
            homog.solve_macro(
                macro_gradients[k],
                temperature_fields[k],
                rtol=args.temperature_rtol,
            )

            # Gradient and flux must be taken from the *periodic* fluctuation:
            # homog.grad is a periodic FE operator, so applying it to a field
            # that already carries the affine ramp puts a spurious spike in
            # the cells that wrap around the cell boundary. The macro gradient
            # is added afterwards instead, giving the total field
            # grad(T) = g + grad(T_fluct).
            homog.engine.communicate_ghosts(temperature_fields[k])
            homog.grad.apply(temperature_fields[k], homog._g)
            g_tot = np.asarray(homog._g.s)
            g_tot = g_tot + macro_gradients[k].reshape(
                (dim,) + (1,) * (g_tot.ndim - 1))
            gradient_fields[k].s[...] = homog.to_device(g_tot)
            flux_fields[k].s[...] = homog.to_device(
                np.asarray(homog.kappa.s) * g_tot)

            # Total temperature = periodic fluctuation + g . x. This makes the
            # stored field non-periodic, so it shows a jump across the cell
            # boundary when plotted -- that ramp is the macro gradient itself.
            # Must come after grad.apply above.
            affine = np.tensordot(macro_gradients[k], positions, axes=(0, 0))
            temperature_fields[k].s[0] += homog.to_device(affine)

        fio.append_frame().write(frame_fields)

        if flush_frames:
            fio.sync()
        frame_iters.append(int(it))

    # Initial configuration as frame 0 (only when dumping intermediate steps).
    if dump_intermediate:
        write_frame(0, rho0)

    stall_seen = [0]

    def cb(it, rho, last):
        vf = comm.sum(float(np.sum(rho))) / n_global
        cg = last.get("cg_iters", [])
        cg_total = int(sum(cg))
        stalled = homog.cg_stagnation_count - stall_seen[0]
        stall_seen[0] = homog.cg_stagnation_count
        hist["objective"].append(float(last["objective"]))
        hist["volume_fraction"].append(vf)
        hist["cg_iters"].append(cg_total)
        if dump_intermediate and it % dump_every == 0:
            write_frame(it, rho)
        if rank0:
            kappa_eff = effective_conductivity(last["fluxes"])
            rtol = last.get("cg_rtol")
            rtol_str = f"  cg-rtol={rtol:.1e}" if rtol is not None else ""
            rtol_str += f"  cg-stalled={stalled}" if stalled else ""
            print(
                f"  bfgs-iter {it:4d}  f={last['objective']:.6e}  "
                f"vol_frac={vf:.3f} "
                f"cg-iters={cg_total}{rtol_str}"
            )
            # Format the array separately first
            kappa_str = np.array2string(
                kappa_eff, formatter={'float_kind': lambda x: f"{x:.4f}"})
            # print(f"Optimized: kappa=\n{kappa_str} (anisotropic)")

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

    homog.set_density(rho)
    homog_data = homog.homogenized_tangent()

    if rank0:
        kappa_eff = effective_conductivity(problem.last["fluxes"])
        print(
            f"done: {info['message']}  f={info['objective']:.6e}  ")
        # Format the array separately first
        kappa_str = np.array2string(
            kappa_eff, formatter={'float_kind': lambda x: f"{x:.4f}"})
        print(f"Optimized: kappa=\n{kappa_str} (anisotropic)")
        homog_data_str = np.array2string(
            homog_data, formatter={'float_kind': lambda x: f"{x:.4f}"})
        print(f"Optimized: homog_data_str=\n{homog_data_str} (anisotropic)")

        if not converged:
            print(
                f"WARNING: L-BFGS did NOT converge; the written density "
                "is the last (non-converged) iterate (converged=0 in the "
                "output file)."
            )

    if args.output is not None:
        # Always include the final iterate as the last frame
        final_it = int(info["nit"])
        if not frame_iters or frame_iters[-1] != final_it:
            write_frame(final_it, rho)

        # Overwrite placeholders with real values
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
        upd("frame_iterations", frame_iters)
        fio.close()
        if rank0:
            print(
                f"wrote {args.output} ({len(frame_iters)} frame(s), "
                f"converged={int(converged)})"
            )
            end = time.time()
            print(f"Elapsed: {end - start:.4f}s")


if __name__ == "__main__":
    main()