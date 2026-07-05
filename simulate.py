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
    python simulate.py -n 96 96 96 --bfgs-iters 300 --eta 0.02
    mpirun -np 4 python simulate.py -n 128 128 128     # (serial optimizer; see notes)

The solve/sensitivity are FFT-accelerated, J-FFT-preconditioned and (with a GPU
build of muGrid + ``--device gpu``) run on device. The outer L-BFGS optimizer is
currently serial.
"""

import argparse

import muGrid
import numpy as np

from muTopOpt import (
    Homogenization,
    NodalPhaseFieldRegularization,
    PhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.loadcases import (
    isotropic_stiffness_from_E_nu,
    isotropic_stiffness_tensor,
    target_load_cases,
)
from muTopOpt.optimize import initial_density, optimize_bounded_lbfgs


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
    p.add_argument(
        "--solid-E", type=float, default=1.0, help="solid Young's modulus"
    )
    p.add_argument(
        "--solid-nu", type=float, default=0.3, help="solid Poisson's ratio"
    )
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
    p.add_argument("--init-volume-fraction", type=float, default=0.5,
                   help="volume fraction of the initial density field")
    p.add_argument(
        "--init",
        choices=["uniform", "random", "filtered_random"],
        default="filtered_random",
        help="initial density field: 'uniform' (constant), 'random' "
        "(white noise) or 'filtered_random' (noise smoothed to a correlation "
        "length; least prone to locking the initial topology)",
    )
    p.add_argument(
        "--init-length",
        type=float,
        default=None,
        help="correlation length for --init filtered_random (default: 3*eta)",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for the --init random/filtered_random "
                        "density field")
    p.add_argument("--bfgs-iters", type=int, default=200,
                   help="maximum number of outer L-BFGS iterations")
    p.add_argument("--output-cg-iters", action="store_true",
                   help="print one line per inner CG iteration (residual and "
                        "relative residual) for every forward/adjoint solve, "
                        "so the CG convergence can be watched live")
    p.add_argument("--bfgs-gtol", type=float, default=1e-5,
                   help="L-BFGS convergence tolerance on the projected gradient")
    p.add_argument("--cg-tol", type=float, default=1e-4,
               help="inner CG relative tolerance; loose values are safe "
                    "because the objective is adjoint-corrected "
                    "(Lagrangian) and thus second-order accurate in the "
                    "solve error")
    p.add_argument("--cg-maxiter", type=int, default=2000,
                   help="maximum number of inner CG iterations per solve "
                        "(the solve stops here even if --cg-tol is not met)")
    p.add_argument("--preconditioner", choices=["green-jacobi", "green"],
                   default="green-jacobi",
                   help="inner-solve preconditioner: 'green-jacobi' (J-FFT, "
                        "reference stiffness times a per-pixel Jacobi scaling) "
                        "or 'green' (plain reference-stiffness Green operator)")
    p.add_argument("--element", choices=["p1", "q1"], default="q1",
                   help="finite element (P1 simplices or Q1 hex/quad)")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="run the forward/adjoint solves and sensitivity on the "
                   "host (cpu) or on the accelerator (gpu); the L-BFGS "
                   "optimizer always runs on the host")
    p.add_argument("--precision", choices=["single", "double"], default="double",
                   help="scalar precision of the on-grid fields and the "
                   "FFT-accelerated solves (the FFT engine dispatches on the "
                   "field dtype); the host L-BFGS optimizer stays double")
    p.add_argument("--density", choices=["element", "nodal"], default="nodal",
                   help="density discretization: 'nodal' (nodal FE field, "
                   "SIMP on the element average of the interpolant, fully "
                   "consistent Galerkin phase-field regularization) or "
                   "'element' (per-pixel density, FD Laplacian penalty)")
    p.add_argument("--output", type=str, default=None,
                   help="NetCDF file to write the optimized density to")
    p.add_argument("--dump-every", type=int, default=-1,
                   help="dump intermediate L-BFGS iterates to the NetCDF output "
                        "as successive frames: with N>0 the initial "
                        "configuration and every N-th iterate (N, 2N, ...) are "
                        "written, and the final iterate is always included; -1 "
                        "(default) writes only the final density as a single "
                        "frame")
    args = p.parse_args()

    dim = len(args.nb_grid_pts)
    if dim not in (2, 3):
        p.error("-n takes 2 or 3 values")
    if args.domain_lengths is not None and len(args.domain_lengths) != dim:
        p.error(f"--domain-lengths must have {dim} values (one per axis)")

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
        tuple(args.nb_grid_pts), material, comm=comm, element=args.element,
        domain_lengths=args.domain_lengths,
        preconditioner=args.preconditioner, cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter, cg_verbose=args.output_cg_iters,
        device=args.device, dtype=dtype,
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
    # The interface width defaults to two grid spacings: wide enough that the
    # regularization can move interfaces (merge/remove features) instead of
    # freezing the initial topology, narrow enough for crisp designs.
    eta = 2.0 * min(homog.grid_spacing) if args.eta is None else args.eta
    Reg = (
        NodalPhaseFieldRegularization
        if args.density == "nodal"
        else PhaseFieldRegularization
    )
    reg = Reg(homog, eta=eta, weight=args.reg_weight)
    problem = StressTargetProblem(homog, cases, regularization=reg, design=args.density)

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

    if rank0:
        print(f"muTopOpt: {dim}D  grid={tuple(args.nb_grid_pts)}  "
              f"load cases={len(cases)}  preconditioner={args.preconditioner}  "
              f"device={args.device}  precision={args.precision}")

    # Per-iteration L-BFGS history, collected across ranks with global
    # reductions so every rank holds the same series (safe to write below).
    n_global = comm.sum(float(rho0.size))
    hist = {"objective": [], "volume_fraction": [], "cg_iters": []}

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
    dump_intermediate = args.output is not None and dump_every is not None \
        and dump_every > 0
    fio = None
    field = None
    fdg_view = None   # numpy view of the per-frame applied-deformation-gradient
    frame_fields = ["density"]  # fields/variables written on every frame
    frame_iters = []  # L-BFGS iteration index of each frame actually written
    _MSG_LEN = 256    # fixed reservation for the optimizer message string
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
            list(applied_deformation_gradient.shape), np.float64,
        )
        fdg_view[...] = applied_deformation_gradient
        frame_fields.append("applied_deformation_gradient")
        # Physical cell size (per-file constant): a global attribute is correct.
        fio.write_global_attribute("domain_lengths",
                                   [float(x) for x in homog.domain_lengths])
        # Placeholders, sized to the maximum they can reach (the optimizer runs
        # at most args.bfgs_iters iterations, hence at most that many history
        # entries and dumped frames). Real values are written after the run.
        maxlen = int(args.bfgs_iters) + 1
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

    def write_frame(it, rho):
        """Stream one density iterate (and the applied deformation gradient) to
        the output as a new frame."""
        field.p[...] = homog.to_device(rho)
        fio.append_frame().write(frame_fields)
        frame_iters.append(int(it))

    # Initial configuration as frame 0 (only when dumping intermediate steps).
    if dump_intermediate:
        write_frame(0, rho0)

    def cb(it, rho, last):
        vf = comm.sum(float(np.sum(rho))) / n_global
        cg = last.get("cg_iters", [])
        cg_total = int(sum(cg))
        hist["objective"].append(float(last["objective"]))
        hist["volume_fraction"].append(vf)
        hist["cg_iters"].append(cg_total)
        if dump_intermediate and it % dump_every == 0:
            write_frame(it, rho)
        if rank0:
            # Per-CG-iteration residuals are reported live during the solves
            # themselves (--output-cg-iters); here we always summarize the
            # outer step and the total inner CG iterations it took (the same
            # total also goes to the NetCDF history above).
            print(f"  bfgs-iter {it:4d}  f={last['objective']:.6e}  "
                  f"vol_frac={vf:.3f}  cg-iters={cg_total}")

    rho, info = optimize_bounded_lbfgs(
        problem, rho0, comm=mpi_comm, maxiter=args.bfgs_iters, gtol=args.bfgs_gtol,
        callback=cb
    )

    converged = bool(info["success"])
    if rank0:
        print(
            f"done: {info['message']}  f={info['objective']:.6e}  iters={info['nit']}"
        )
        if not converged:
            print(
                "WARNING: L-BFGS did NOT converge; the written density is "
                "the last (non-converged) iterate (converged=0 in the "
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
        upd("frame_iterations", frame_iters)
        fio.close()
        if rank0:
            print(f"wrote {args.output} ({len(frame_iters)} frame(s), "
                  f"converged={int(converged)})")


if __name__ == "__main__":
    main()
