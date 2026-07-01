#!/usr/bin/env python3
#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Command-line driver for muTopOpt: optimize an element-wise density unit cell for
a target isotropic effective stiffness (bulk modulus K, shear modulus G), in 2D
or 3D.

Examples
--------
    python simulate.py -n 64 64            --K 0.1 --G 0.05
    python simulate.py -n 96 96 96 --iters 300 --eta 2.0
    mpirun -np 4 python simulate.py -n 128 128 128     # (serial optimizer; see notes)

The solve/sensitivity are FFT-accelerated, J-FFT-preconditioned and (with a GPU
build of muGrid + ``--device gpu``) run on device. The outer L-BFGS optimizer is
currently serial.
"""

import argparse

import numpy as np

import muGrid
from muTopOpt import (Homogenization, NodalPhaseFieldRegularization,
                      PhaseFieldRegularization, SimpMaterial,
                      StressTargetProblem)
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases
from muTopOpt.optimize import initial_density, optimize_bounded_lbfgs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--nb-grid-pts", type=int, nargs="+", required=True,
                   help="grid points per axis (2 or 3 values)")
    p.add_argument("--E", type=float, default=1.0, help="solid Young's modulus")
    p.add_argument("--nu", type=float, default=0.3, help="Poisson's ratio")
    p.add_argument("--penalty", type=float, default=3.0, help="SIMP exponent p")
    p.add_argument("--void-ratio", type=float, default=1e-3,
                   help="void/solid stiffness ratio")
    p.add_argument("--K", type=float, default=0.1, help="target bulk modulus")
    p.add_argument("--G", type=float, default=0.05, help="target shear modulus")
    p.add_argument("--eta", type=float, default=1.0,
                   help="phase-field interface-width parameter")
    p.add_argument("--well", type=float, default=1e-3,
                   help="double-well weight")
    p.add_argument("--volume-fraction", type=float, default=0.5)
    p.add_argument("--init", choices=["uniform", "random"], default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--cg-tol", type=float, default=1e-8)
    p.add_argument("--preconditioner", choices=["green-jacobi", "green"],
                   default="green-jacobi")
    p.add_argument("--element", choices=["p1", "q1"], default="q1",
                   help="finite element (P1 simplices or Q1 hex/quad)")
    p.add_argument("--density", choices=["element", "nodal"], default="element",
                   help="density discretization: 'element' (per-pixel, FD "
                   "Laplacian penalty) or 'nodal' (nodal FE field with the "
                   "element-consistent, fused FE-Laplacian penalty)")
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
    homog = Homogenization(
        tuple(args.nb_grid_pts), material, comm=comm, element=args.element,
        preconditioner=args.preconditioner, cg_tol=args.cg_tol,
    )
    cases = target_load_cases(
        dim, isotropic_stiffness_tensor(dim, args.K, args.G), magnitude=0.01
    )
    Reg = (NodalPhaseFieldRegularization if args.density == "nodal"
           else PhaseFieldRegularization)
    reg = Reg(homog, eta=args.eta, well_weight=args.well)
    problem = StressTargetProblem(homog, cases, regularization=reg)

    rho0 = initial_density(
        homog.nb_pixels, kind=args.init,
        volume_fraction=args.volume_fraction, seed=args.seed,
    )

    if rank0:
        print(f"muTopOpt: {dim}D  grid={tuple(args.nb_grid_pts)}  "
              f"load cases={len(cases)}  preconditioner={args.preconditioner}")

    def cb(it, rho, last):
        if rank0:
            vf = float(np.mean(rho))
            print(f"  iter {it:4d}  f={last['objective']:.6e}  vol_frac={vf:.3f}")

    rho, info = optimize_bounded_lbfgs(
        problem, rho0, comm=mpi_comm, maxiter=args.iters, callback=cb
    )

    if rank0:
        print(f"done: {info['message']}  f={info['objective']:.6e}  "
              f"iters={info['nit']}")

    if args.output is not None:
        field = homog.scalar_field("density")
        field.p[...] = rho
        fio = muGrid.FileIONetCDF(
            args.output, muGrid.FileIONetCDF.OpenMode.Overwrite, comm
        )
        fio.register_field_collection(homog.fc)
        fio.append_frame().write(["density"])
        if rank0:
            print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
