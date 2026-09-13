#!/usr/bin/env python3
#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""Benchmark the inner state solve -- the hot loop of a muTopOpt optimization.

An optimization step spends nearly all its time in preconditioned CG solves
(one per load case for the state, two more per Hessian-vector product), so the
cost of a single CG iteration is what determines the run time. This driver
measures exactly that, on a fixed, reproducible design, without the outer
optimizer in the way.

The headline number is **milliseconds per CG iteration**. It separates the two
things an optimization can change:

* *time per CG iteration* -- an implementation change (kernels, memory layout,
  launch overhead). It should improve without moving the iteration count.
* *iterations per solve* -- a numerical change (preconditioner, tolerances,
  precision). Report both, and treat a change in the second as something that
  needs justifying, not a speedup.

Usage::

    python benchmarks/solve_bench.py -n 64                    # GPU, float32
    python benchmarks/solve_bench.py -n 64 --device cpu
    python benchmarks/solve_bench.py -n 96 --precision double
    python benchmarks/solve_bench.py -n 64 --repeat 3         # spread of runs
    python benchmarks/solve_bench.py -n 64 --csv results.csv  # append a row

Comparing two builds means running the same command against each and comparing
`ms/CG-iter`. Run-to-run spread on a laptop GPU is a few percent, so use
``--repeat`` before believing a small difference.
"""

import argparse
import csv
import os
import platform
import time

import numpy as np

from muTopOpt import Homogenization, SimpMaterial
from muTopOpt.optimize import initial_density

#: Fields written by ``--csv``, in order.
CSV_COLUMNS = (
    "timestamp", "host", "device", "precision", "dim", "n", "nb_pixels",
    "preconditioner", "element", "rtol", "solves", "iters_per_solve",
    "ms_per_solve", "ms_per_cg_iter",
)


def build(args):
    """Construct the homogenization problem and apply a fixed design."""
    dtype = {"single": np.float32, "double": np.float64}[args.precision]
    material = SimpMaterial(
        E_solid=args.solid_E, nu=args.solid_nu,
        penalty=args.penalty, void_ratio=args.void_ratio,
    )
    homog = Homogenization(
        (args.n,) * args.dim, material,
        element=args.element, preconditioner=args.preconditioner,
        device=args.device, dtype=dtype,
    )
    # The same smooth random design simulate.py starts from (correlation
    # length 3*eta, eta = one grid spacing), so the material contrast the
    # preconditioner sees is representative rather than uniform.
    rho = initial_density(
        homog.nb_pixels, kind="filtered_random", seed=args.seed,
        length=3.0 * float(homog.grid_spacing[0]),
        grid_spacing=homog.grid_spacing,
    )
    homog.set_density(rho)
    return homog


def unit_strains(dim):
    """One unit macro strain per independent direction of the symmetric strain
    space: `dim` normal directions followed by `dim*(dim-1)/2` shears."""
    cases = []
    for i in range(dim):
        E = np.zeros((dim, dim))
        E[i, i] = 1.0
        cases.append(E)
    for i in range(dim):
        for j in range(i + 1, dim):
            E = np.zeros((dim, dim))
            E[i, j] = E[j, i] = 0.5
            cases.append(E)
    return cases


def run(homog, strains, nb_solves, rtol, warmup):
    """Time `nb_solves` state solves, cycling through the load cases.

    Returns (seconds per solve, CG iterations per solve).
    """
    x = homog.vector_field("solve-bench-u")

    def one(k):
        # Cold start every solve: a warm start would make the timing depend on
        # the order the load cases happen to be visited in.
        x.set_zero()
        homog.solve_macro(strains[k % len(strains)], x, rtol=rtol)

    def sync():
        if homog.on_device:
            homog._xp.cuda.runtime.deviceSynchronize()

    for k in range(warmup):
        one(k)
    sync()

    times, iters = [], []
    for k in range(nb_solves):
        t0 = time.perf_counter()
        one(k)
        sync()
        times.append(time.perf_counter() - t0)
        iters.append(homog.last_cg_iters)
    return times, iters


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-n", type=int, default=64,
                   help="grid points per axis (default: 64)")
    p.add_argument("--dim", type=int, default=3, choices=(2, 3),
                   help="spatial dimension (default: 3)")
    p.add_argument("--device", default="gpu",
                   help="'cpu', 'gpu', 'cuda:N', ... (default: gpu)")
    p.add_argument("--precision", default="single",
                   choices=("single", "double"),
                   help="solver precision (default: single)")
    p.add_argument("--preconditioner", default="green-jacobi",
                   choices=("green-jacobi", "green"))
    p.add_argument("--element", default="p1", choices=("p1", "q1"))
    p.add_argument("--rtol", type=float, default=1e-2,
                   help="inner CG relative tolerance. The default matches the "
                        "loose tolerance the optimizer starts from, which is "
                        "where most iterations are spent (default: 1e-2)")
    p.add_argument("--solves", type=int, default=4,
                   help="timed solves (default: 4)")
    p.add_argument("--warmup", type=int, default=1,
                   help="untimed solves first, to pay one-off costs such as "
                        "FFT planning and allocation (default: 1)")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat the whole measurement, to show the spread "
                        "(default: 1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solid-E", dest="solid_E", type=float, default=1.0)
    p.add_argument("--solid-nu", dest="solid_nu", type=float, default=0.3)
    p.add_argument("--penalty", type=float, default=2.0)
    p.add_argument("--void-ratio", dest="void_ratio", type=float, default=1e-3)
    p.add_argument("--csv", metavar="FILE",
                   help="append one row per repeat to FILE (header written "
                        "when the file is new)")
    args = p.parse_args()

    homog = build(args)
    strains = unit_strains(args.dim)

    print(f"n={args.n}^{args.dim}  device={args.device}  "
          f"precision={args.precision}  preconditioner={args.preconditioner}  "
          f"element={args.element}  rtol={args.rtol:g}", flush=True)

    rows = []
    for rep in range(args.repeat):
        times, iters = run(homog, strains, args.solves, args.rtol, args.warmup)
        tot_t, tot_i = sum(times), sum(iters)
        per_solve = tot_t / args.solves * 1e3
        per_iter = tot_t / max(tot_i, 1) * 1e3
        print(f"  iters/solve {np.mean(iters):7.1f}   "
              f"ms/solve {per_solve:8.1f}   "
              f"ms/CG-iter {per_iter:7.3f}", flush=True)
        rows.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": platform.node(),
            "device": args.device,
            "precision": args.precision,
            "dim": args.dim,
            "n": args.n,
            "nb_pixels": int(np.prod(homog.nb_pixels)),
            "preconditioner": args.preconditioner,
            "element": args.element,
            "rtol": args.rtol,
            "solves": args.solves,
            "iters_per_solve": round(float(np.mean(iters)), 2),
            "ms_per_solve": round(per_solve, 3),
            "ms_per_cg_iter": round(per_iter, 4),
        })

    if args.repeat > 1:
        vals = [r["ms_per_cg_iter"] for r in rows]
        print(f"  ms/CG-iter over {args.repeat} repeats: "
              f"min {min(vals):.3f}  median {np.median(vals):.3f}  "
              f"max {max(vals):.3f}")

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"  appended {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
