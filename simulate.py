#!/usr/bin/env python3
#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Command-line driver for muTopOpt: optimize a density unit cell for a target
isotropic effective stiffness, in 2D or 3D. The target is given either as bulk
and shear moduli (``--K``/``--G``) or as Young's modulus and Poisson's ratio
(``--target-E``/``--target-nu``).

By default the density is a *nodal* finite-element field with the fully
consistent Galerkin phase-field regularization, started from a filtered random
field -- the combination least prone to locking in the initial topology.

Examples
--------
    python simulate.py -n 64 64            --K 0.1 --G 0.05
    python simulate.py -n 64 64            --target-E 0.2 --target-nu -0.3
    python simulate.py -n 96 96 96 --iters 300 --eta 0.02
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
    p.add_argument("--E", type=float, default=1.0, help="solid Young's modulus")
    p.add_argument("--nu", type=float, default=0.3, help="Poisson's ratio")
    p.add_argument("--penalty", type=float, default=2.0, help="SIMP exponent p")
    p.add_argument(
        "--void-ratio", type=float, default=1e-3, help="void/solid stiffness ratio"
    )
    p.add_argument("--K", type=float, default=0.1, help="target bulk modulus")
    p.add_argument("--G", type=float, default=0.05, help="target shear modulus")
    p.add_argument(
        "--target-E",
        type=float,
        default=None,
        help="target Young's modulus (alternative to --K/--G; requires --target-nu)",
    )
    p.add_argument(
        "--target-nu",
        type=float,
        default=None,
        help="target Poisson's ratio (alternative to --K/--G; requires --target-E)",
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
    p.add_argument("--volume-fraction", type=float, default=0.5)
    p.add_argument(
        "--init",
        choices=["uniform", "random", "filtered_random"],
        default="filtered_random",
    )
    p.add_argument(
        "--init-length",
        type=float,
        default=None,
        help="correlation length for --init filtered_random (default: 3*eta)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--gtol", type=float, default=1e-5,
                   help="L-BFGS convergence tolerance on the projected gradient")
    p.add_argument("--cg-tol", type=float, default=1e-4,
               help="inner CG relative tolerance; loose values are safe "
                    "because the objective is adjoint-corrected "
                    "(Lagrangian) and thus second-order accurate in the "
                    "solve error")
    p.add_argument("--preconditioner", choices=["green-jacobi", "green"],
                   default="green-jacobi")
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
    args = p.parse_args()

    dim = len(args.nb_grid_pts)
    if dim not in (2, 3):
        p.error("-n takes 2 or 3 values")

    if muGrid.has_mpi:
        from mpi4py import MPI

        mpi_comm = MPI.COMM_WORLD
        comm = muGrid.Communicator(mpi_comm)
    else:
        mpi_comm = None
        comm = muGrid.Communicator()
    rank0 = comm.rank == 0

    material = SimpMaterial(args.E, args.nu, args.penalty, args.void_ratio)
    # CLI exposes "single"/"double"; muTopOpt works in NumPy dtypes internally.
    dtype = {"single": np.float32, "double": np.float64}[args.precision]
    homog = Homogenization(
        tuple(args.nb_grid_pts), material, comm=comm, element=args.element,
        preconditioner=args.preconditioner, cg_tol=args.cg_tol,
        device=args.device, dtype=dtype,
    )
    if (args.target_E is None) != (args.target_nu is None):
        p.error("--target-E and --target-nu must be given together")
    if args.target_E is not None:
        target = isotropic_stiffness_from_E_nu(dim, args.target_E, args.target_nu)
    else:
        target = isotropic_stiffness_tensor(dim, args.K, args.G)
    cases = target_load_cases(dim, target, magnitude=0.01)
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
        volume_fraction=args.volume_fraction,
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

    def cb(it, rho, last):
        vf = comm.sum(float(np.sum(rho))) / n_global
        cg = last.get("cg_iters", [])
        cg_total = int(sum(cg))
        hist["objective"].append(float(last["objective"]))
        hist["volume_fraction"].append(vf)
        hist["cg_iters"].append(cg_total)
        if rank0:
            cg_str = ""
            if cg:
                # cg is [fwd1, adj1, fwd2, adj2, ...]: one (forward, adjoint)
                # solve pair per load case. Cluster into (fwd, adj) tuples.
                pairs = ",".join(f"({f},{a})"
                                 for f, a in zip(cg[0::2], cg[1::2]))
                cg_str = f"  cg_iters=[{pairs}] (total {cg_total})"
            print(f"  iter {it:4d}  f={last['objective']:.6e}  "
                  f"vol_frac={vf:.3f}{cg_str}")

    rho, info = optimize_bounded_lbfgs(
        problem, rho0, comm=mpi_comm, maxiter=args.iters, gtol=args.gtol, callback=cb
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
        field = homog.scalar_field("density")
        field.p[...] = homog.to_device(rho)
        fio = muGrid.FileIONetCDF(
            args.output, muGrid.FileIONetCDF.OpenMode.Overwrite, comm
        )
        fio.register_field_collection(homog.fc)
        # Global attributes (incl. the L-BFGS history, one value per outer
        # iteration, for later plotting). These MUST be written before any
        # field data -- muGrid forbids growing the header once a frame has
        # been written. All quantities are globally consistent across ranks
        # (NuMPI's l_bfgs_bounded returns the same result on every rank), so
        # writing them from all ranks is safe.
        # Convergence status: `converged` is the machine-readable flag (1 =
        # the optimizer met its tolerances, 0 = it stopped early, e.g. at
        # maxiter); the remaining attributes give the reason and the final
        # optimizer state.
        fio.write_global_attribute("converged", [int(converged)])
        fio.write_global_attribute("optimizer_message", str(info["message"]))
        fio.write_global_attribute("nb_iterations", [int(info["nit"])])
        fio.write_global_attribute("final_objective", [float(info["objective"])])
        fio.write_global_attribute("final_max_gradient", [float(info["max_grad"])])
        if hist["objective"]:
            fio.write_global_attribute("lbfgs_objective_history", hist["objective"])
            fio.write_global_attribute(
                "lbfgs_volume_fraction_history", hist["volume_fraction"]
            )
            fio.write_global_attribute("lbfgs_cg_iters_history", hist["cg_iters"])
        fio.append_frame().write(["density"])
        if rank0:
            print(f"wrote {args.output} (converged={int(converged)})")


if __name__ == "__main__":
    main()
