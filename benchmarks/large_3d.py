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
  shear modulus 0.3 · 0.15 E_solid (9 energies, 6 load cases), or with
  ``--objective stress`` the stresses of the 6 load cases matched to the
  target's;
* phase-field regularization with weight 0.2, interface width ``η = 2h``
  and the lumped double well; the gradient term with one quadrature point
  per voxel (``--pf-quadrature one-point``) is the study's, the default
  ``full`` is the exact integral of the interpolant.

Every option that makes the solve fast can be switched, to measure what it
buys:

* ``--preconditioner multigrid | green-jacobi``, and
  ``--multigrid-coarsening galerkin | rediscretised``
* ``--cg-stop energy | residual``
* ``--warm-start device | device32 | host | host32 | none``
* ``--optimizer helmholtz | numpi`` (L-BFGS on fields with the Helmholtz
  smoother, or NuMPI's L-BFGS on host arrays)
* ``--lbfgs-pairs device | host`` (where the L-BFGS on fields keeps its
  curvature pairs)

Grid continuation. Several grids run one after the other, each starting
from the final design of the previous one, interpolated trilinearly
(:func:`muTopOpt.restart.linear_resample`); the interface width stays two
voxels on every grid. Each grid writes its final design to
``<output>_N<n>.nc``, which is also how the next grid receives it, and
which ``--start`` of a later run reads (a chain over several jobs)::

    python benchmarks/large_3d.py --nb-grid-pts 32 64 128 --output chain \\
        --start CH3D_start_bi_s11.nc --maxiter 200

``--start`` takes such a muTopOpt file or a muFFTTO file (variable
``phase_field_solution``, e.g. the Cahn-Hilliard starts of the study), on
any grid.

One line is printed per evaluation, with the objective and the CG
iterations of the forward and adjoint solves, and a summary per grid. With
``--memory`` the summary also gives the peak device memory of the process
(largest over the ranks) and its bytes per voxel of the rank's subdomain.
"""
import argparse
import gc
import os
import shlex
import subprocess
import sys
import threading
import time

import numpy as np

import muGrid

from muTopOpt import (
    Homogenization,
    NodalPhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.loadcases import (
    auxetic_target,
    energy_load_cases,
    target_load_cases,
)
from muTopOpt.optimize import optimize_bounded_lbfgs, optimize_helmholtz_lbfgs
from muTopOpt.restart import linear_resample, read_final_density, write_density


def read_start(path):
    """The start design on its own grid, as a global array: from a muTopOpt
    file (:func:`muTopOpt.restart.write_density` or ``simulate.py``, which
    carry the attribute ``nb_grid_pts``), from a muFFTTO file (variable
    ``phase_field_solution`` of a final design or a Cahn-Hilliard start,
    ``phase_field_iterate`` of a checkpoint; last frame), or without a file
    a smooth random field (8³ values, interpolated later)."""
    if path is None:
        return np.random.default_rng(11).uniform(0.0, 1.0, (8, 8, 8))
    fio = muGrid.FileIONetCDF(path, muGrid.FileIONetCDF.OpenMode.Read,
                              muGrid.Communicator())
    try:
        written_by_mutopopt = "nb_grid_pts" in fio.read_global_attribute_names()
    finally:
        fio.close()
    if written_by_mutopopt:
        return read_final_density(path)[0]
    import netCDF4

    with netCDF4.Dataset(path) as data:
        name = next(name for name in ("phase_field_solution",
                                      "phase_field_iterate")
                    if name in data.variables)
        return np.asarray(data[name][-1, 0], dtype=float)


def release_device_memory():
    """Return the memory of a finished grid to the device: the fields go
    with their collection once nothing refers to it any more, and cupy keeps
    freed blocks in its pools until asked."""
    gc.collect()
    try:
        import cupy
    except ImportError:
        return
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


class DeviceMemoryPeak:
    """Peak device memory of this process, sampled with ``nvidia-smi`` every
    ``interval`` seconds in a background thread while the context is open.

    ``nvidia-smi`` reports what the driver gave the process: muGrid's
    fields, cupy's memory pools and the CUDA context (about 100–200 MiB);
    other processes on the same GPU do not count. A peak shorter than the
    interval can be missed. Without ``nvidia-smi`` the peak stays 0."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_mib = 0
        self._running = False
        self._thread = None

    def sample(self):
        """This process's device memory now, in MiB (0 if unknown)."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            return 0
        for line in out.splitlines():
            pid, mib = (int(value) for value in line.split(","))
            if pid == os.getpid():
                return mib
        return 0

    def _run(self):
        while self._running:
            self.peak_mib = max(self.peak_mib, self.sample())
            time.sleep(self.interval)

    def __enter__(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._running = False
        self._thread.join()
        self.peak_mib = max(self.peak_mib, self.sample())


def run_grid(n, start, args, comm):
    """Optimise on the ``n³`` grid from the design ``start`` (global array on
    any grid). Returns the summary line of this grid; with ``--output`` the
    final design is written to ``<output>_N<n>.nc``."""
    if args.memory:
        with DeviceMemoryPeak() as memory:
            summary, voxels = _run_grid(n, start, args, comm)
        # bytes per voxel of each rank's own subdomain; the largest rank
        per_voxel = comm.max(memory.peak_mib * 2**20 / voxels)
        summary += (f", peak device memory {comm.max(memory.peak_mib)} MiB "
                    f"per rank, {per_voxel:.0f} B/voxel (incl. the CUDA "
                    f"context)")
    else:
        summary, _ = _run_grid(n, start, args, comm)
    if comm.rank == 0:
        print(summary, flush=True)
    return summary


def _run_grid(n, start, args, comm):
    """The work of :func:`run_grid`; returns the summary line and the
    number of voxels of this rank."""
    rank0 = comm.rank == 0
    started = time.time()

    # Solid with bulk modulus 1 and shear modulus 0.5
    bulk, shear = 1.0, 0.5
    E_solid = 9 * bulk * shear / (3 * bulk + shear)
    nu_solid = (3 * bulk - 2 * shear) / (2 * (3 * bulk + shear))
    material = SimpMaterial(E_solid, nu_solid, penalty=2.0, void_ratio=1e-3)
    homogenization = Homogenization(
        (n, n, n), material, comm=comm, device=args.device,
        preconditioner=args.preconditioner, cg_tol=args.cg_tol,
        cg_stop=args.cg_stop, multigrid_precision=args.multigrid_precision,
        multigrid_coarsening=args.multigrid_coarsening)
    regularization = NodalPhaseFieldRegularization(
        homogenization, eta=2.0 / n, weight=args.pf_weight, dwell="lumped",
        gradient_quadrature=args.pf_quadrature.replace("-", "_"))
    target = auxetic_target(3, E_solid, nu=args.target_nu,
                            scale=args.target_scale)
    cases = (energy_load_cases(3, target) if args.objective == "energy"
             else target_load_cases(3, target))
    problem = StressTargetProblem(
        homogenization, cases, regularization=regularization, design="nodal",
        warm_start=None if args.warm_start == "none" else args.warm_start)

    # count and time the evaluations
    evaluate = problem.objective_and_gradient
    totals = {"evaluations": 0, "forward": 0, "adjoint": 0, "seconds": 0.0}

    def timed_evaluate(rho):
        start_evaluation = time.time()
        f, g = evaluate(rho)
        seconds = time.time() - start_evaluation
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

    # the start on this rank's nodes; clipped, since a start from another
    # code (e.g. a Cahn-Hilliard field) may leave [0, 1] slightly
    rho0 = np.clip(linear_resample(
        start, (n, n, n), homogenization.engine.subdomain_locations,
        homogenization.nb_pixels), 0.0, 1.0)
    if rank0:
        print(f"3D TO {n}^3 on {comm.size} rank(s), device {args.device}, "
              f"preconditioner {args.preconditioner} "
              f"({args.multigrid_coarsening}), CG stop "
              f"{args.cg_stop} at {args.cg_tol:g}, warm starts "
              f"{args.warm_start}, optimizer {args.optimizer} (pairs on "
              f"{args.lbfgs_pairs}), {args.objective} objective, gradient term "
              f"{args.pf_quadrature}, start {start.shape[0]}^3", flush=True)
    options = dict(maxiter=args.maxiter, gtol=0.0, xtol=1e-3, maxcor=20)
    if args.optimizer == "helmholtz":
        rho, info = optimize_helmholtz_lbfgs(problem, rho0,
                                             pair_store=args.lbfgs_pairs,
                                             **options)
    else:
        rho, info = optimize_bounded_lbfgs(problem, rho0, comm=comm,
                                           **options)

    if args.output is not None:
        write_density(output_name(args.output, n), homogenization, rho, {
            "objective": [float(info["objective"])],
            "nb_iterations": [int(info["nit"])],
            "optimizer_message": str(info["message"]),
            "command_line": shlex.join(sys.argv)})

    summary = (f"{n}^3: {info['message']} after {info['nit']} iterations, "
               f"objective {info['objective']:.12e}, wall "
               f"{time.time() - started:.1f} s, evaluations "
               f"{totals['evaluations']} in {totals['seconds']:.1f} s, CG "
               f"forward {totals['forward']}, adjoint {totals['adjoint']}")
    return summary, int(np.prod(homogenization.nb_pixels))


def output_name(prefix, n):
    return f"{prefix}_N{n}.nc"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nb-grid-pts", type=int, nargs="+", default=[32],
                        help="grid, or several grids run coarse to fine")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maxiter", type=int, default=30,
                        help="L-BFGS iterations per grid")
    parser.add_argument("--start", default=None,
                        help="NetCDF file of a nodal start design (muTopOpt "
                             "or muFFTTO), on any grid")
    parser.add_argument("--output", default=None,
                        help="prefix of the final designs <output>_N<n>.nc; "
                             "needed for several grids")
    parser.add_argument("--preconditioner", default="multigrid")
    parser.add_argument("--multigrid-precision", default="single")
    parser.add_argument("--multigrid-coarsening", default="galerkin",
                        choices=("galerkin", "rediscretised"))
    parser.add_argument("--cg-stop", default="energy")
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--warm-start", default="device")
    parser.add_argument("--optimizer", default="helmholtz")
    parser.add_argument("--lbfgs-pairs", default="device",
                        choices=("device", "host"),
                        help="where the L-BFGS on fields keeps its pairs")
    parser.add_argument("--objective", default="energy",
                        choices=("energy", "stress"))
    parser.add_argument("--target-nu", type=float, default=-0.5)
    parser.add_argument("--target-scale", type=float, default=0.3)
    parser.add_argument("--pf-weight", type=float, default=0.2)
    parser.add_argument("--memory", action="store_true",
                        help="report the peak device memory per grid "
                             "(nvidia-smi)")
    parser.add_argument("--pf-quadrature", default="full",
                        choices=("full", "one-point"),
                        help="quadrature of the phase-field gradient term")
    args = parser.parse_args()
    if len(args.nb_grid_pts) > 1 and args.output is None:
        parser.error("several grids need --output: each grid hands its "
                     "design to the next through its output file")

    comm = muGrid.Communicator.world()
    started = time.time()
    start = read_start(args.start)
    summaries = []
    for stage, n in enumerate(args.nb_grid_pts):
        if stage > 0:
            # the previous grid's final design, as the next job would read it
            start = read_final_density(
                output_name(args.output, args.nb_grid_pts[stage - 1]))[0]
        summaries.append(run_grid(n, start, args, comm))
        release_device_memory()

    if comm.rank == 0:
        if len(summaries) > 1:
            print("CHAIN\n  " + "\n  ".join(summaries))
        print(f"TOTAL wall {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
