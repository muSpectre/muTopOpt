#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
The 3D auxetic design problem of the large-grid studies, in muTopOpt.

    python benchmarks/large_3d.py --nb-grid-pts 64 --device gpu --maxiter 30

Problem (the reference setup of the muFFTTO study ``to3d_large.py``):

* periodic unit cell, trilinear hexahedra, nodal density;
* material ``C(ρ) = [k + ρ_e² (1 − k)] C_solid`` with ``k = 1e-3`` and
  ``ρ_e`` the element average of the density;
* energy objective for an isotropic target with Poisson ratio −0.5 and
  shear modulus 0.3 · 0.15 E_solid (9 energies, 6 load cases);
* phase-field regularization with weight 0.2, interface width ``η = 2h``
  and the lumped double well; the gradient term with one quadrature point
  per voxel (``--pf-quadrature one-point``) is the study's, the default
  ``full`` is the exact integral of the interpolant.

Every option that makes the solve fast can be switched, to measure what it
buys:

* ``--preconditioner multigrid | green-jacobi``
* ``--cg-stop energy | residual``
* ``--warm-start device | device32 | host | host32 | none``
* ``--optimizer helmholtz | numpi`` (L-BFGS on fields with the Helmholtz
  smoother, or NuMPI's L-BFGS on host arrays)

One line is printed per evaluation, with the objective and the CG
iterations of the forward and adjoint solves, and a summary at the end.
"""
import argparse
import time

import numpy as np

import muGrid

from muTopOpt import (
    Homogenization,
    NodalPhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.loadcases import auxetic_target, energy_load_cases
from muTopOpt.optimize import optimize_bounded_lbfgs, optimize_helmholtz_lbfgs


def periodic_interpolation(values, homogenization):
    """Trilinear interpolation of a periodic nodal field given on a cubic
    grid onto this process's nodes, axis by axis."""
    coarse = values.shape[0]
    locations = homogenization.engine.subdomain_locations
    for axis, (location, size, n) in enumerate(zip(
            locations, homogenization.nb_pixels,
            homogenization.nb_grid_pts)):
        position = (int(location) + np.arange(int(size))) * (coarse / n)
        base = np.floor(position).astype(int)
        fraction = (position - base).reshape(
            [-1 if a == axis else 1 for a in range(3)])
        values = (np.take(values, base % coarse, axis=axis) * (1 - fraction)
                  + np.take(values, (base + 1) % coarse, axis=axis) * fraction)
    return np.clip(values, 0.0, 1.0)


def start_design(path, homogenization):
    """The start: a saved nodal design (NetCDF, variable
    ``phase_field_solution``), or without a file a smooth random field."""
    if path is None:
        rng = np.random.default_rng(11)
        coarse = rng.uniform(0.0, 1.0, (8, 8, 8))
    else:
        import netCDF4

        with netCDF4.Dataset(path) as data:
            coarse = np.asarray(data["phase_field_solution"][-1, 0])
    return periodic_interpolation(coarse, homogenization)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nb-grid-pts", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--start", default=None,
                        help="NetCDF file of a nodal start design")
    parser.add_argument("--preconditioner", default="multigrid")
    parser.add_argument("--multigrid-precision", default="single")
    parser.add_argument("--cg-stop", default="energy")
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--warm-start", default="device")
    parser.add_argument("--optimizer", default="helmholtz")
    parser.add_argument("--target-nu", type=float, default=-0.5)
    parser.add_argument("--target-scale", type=float, default=0.3)
    parser.add_argument("--pf-weight", type=float, default=0.2)
    parser.add_argument("--pf-quadrature", default="full",
                        choices=("full", "one-point"),
                        help="quadrature of the phase-field gradient term")
    args = parser.parse_args()

    comm = muGrid.Communicator.world()
    rank0 = comm.rank == 0
    n = args.nb_grid_pts
    started = time.time()

    # Solid with bulk modulus 1 and shear modulus 0.5
    bulk, shear = 1.0, 0.5
    E_solid = 9 * bulk * shear / (3 * bulk + shear)
    nu_solid = (3 * bulk - 2 * shear) / (2 * (3 * bulk + shear))
    material = SimpMaterial(E_solid, nu_solid, penalty=2.0, void_ratio=1e-3)
    homogenization = Homogenization(
        (n, n, n), material, comm=comm, device=args.device,
        preconditioner=args.preconditioner, cg_tol=args.cg_tol,
        cg_stop=args.cg_stop, multigrid_precision=args.multigrid_precision)
    regularization = NodalPhaseFieldRegularization(
        homogenization, eta=2.0 / n, weight=args.pf_weight, dwell="lumped",
        gradient_quadrature=args.pf_quadrature.replace("-", "_"))
    cases = energy_load_cases(3, auxetic_target(
        3, E_solid, nu=args.target_nu, scale=args.target_scale))
    problem = StressTargetProblem(
        homogenization, cases, regularization=regularization, design="nodal",
        warm_start=None if args.warm_start == "none" else args.warm_start)

    # count and time the evaluations
    evaluate = problem.objective_and_gradient
    totals = {"evaluations": 0, "forward": 0, "adjoint": 0, "seconds": 0.0}

    def timed_evaluate(rho):
        start = time.time()
        f, g = evaluate(rho)
        seconds = time.time() - start
        iterations = problem.last["cg_iters"]
        totals["evaluations"] += 1
        totals["forward"] += sum(iterations[0::2])
        totals["adjoint"] += sum(iterations[1::2])
        totals["seconds"] += seconds
        if rank0:
            print(f"EVAL {totals['evaluations']} f {f:.12e} wall "
                  f"{time.time() - started:.2f} eval {seconds:.2f} s  CG "
                  f"forward {iterations[0::2]} adjoint {iterations[1::2]}",
                  flush=True)
        return f, g

    problem.objective_and_gradient = timed_evaluate

    rho0 = start_design(args.start, homogenization)
    if rank0:
        print(f"3D TO {n}^3 on {comm.size} rank(s), device {args.device}, "
              f"preconditioner {args.preconditioner}, CG stop "
              f"{args.cg_stop} at {args.cg_tol:g}, warm starts "
              f"{args.warm_start}, optimizer {args.optimizer}, gradient term "
              f"{args.pf_quadrature}", flush=True)
    options = dict(maxiter=args.maxiter, gtol=0.0, xtol=1e-3, maxcor=20)
    if args.optimizer == "helmholtz":
        rho, info = optimize_helmholtz_lbfgs(problem, rho0, **options)
    else:
        rho, info = optimize_bounded_lbfgs(problem, rho0, comm=comm,
                                           **options)

    if rank0:
        print(f"{info['message']} after {info['nit']} iterations, objective "
              f"{info['objective']:.12e}")
        print(f"TOTAL wall {time.time() - started:.1f} s, evaluations "
              f"{totals['evaluations']} in {totals['seconds']:.1f} s, CG "
              f"forward {totals['forward']}, adjoint {totals['adjoint']}")


if __name__ == "__main__":
    main()
