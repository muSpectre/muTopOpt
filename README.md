# muTopOpt

FFT-accelerated finite-element **topology optimization** of mechanical
metamaterials, built on [muGrid](https://github.com/muSpectre/muGrid).

muTopOpt designs a periodic unit cell whose homogenized (effective) stiffness
matches a prescribed target — e.g. a given bulk/shear modulus or a negative
Poisson's ratio. It uses a SIMP material interpolation, a stress-matching
objective with Modica–Mortola phase-field regularization (no explicit volume
constraint), and the discrete adjoint method for exact sensitivities. The
micromechanical equilibrium and the adjoint problem — the two expensive parts
of every optimization step — are solved by muGrid's matrix-free, GPU-capable,
J-FFT (Green–Jacobi) preconditioned conjugate gradient. Unit cells can be 2D
or 3D, the density can live on elements or on nodes, and runs can be
MPI-parallel. A heat-conduction (scalar diffusion) variant is included.

The mathematics is documented in the module docstrings:

| Topic | Module |
|---|---|
| SIMP interpolation $C(\rho) = C_\text{void} + \rho^p (C_\text{solid} - C_\text{void})$ | `muTopOpt/material.py` |
| Cell problem $K u = -B^T C : \bar E$, homogenized stress, CG safeguards, GPU memory | `muTopOpt/homogenization.py` |
| Macro-strain load cases, isotropic targets | `muTopOpt/loadcases.py` |
| Objective, adjoint gradient, adjoint-corrected Lagrangian, Hessian-vector products | `muTopOpt/problem.py` |
| Modica–Mortola regularization | `muTopOpt/regularization.py` |
| Nodal design field, exact double-well integration | `muTopOpt/nodal.py` |
| Voigt / Hashin–Shtrikman feasibility bounds | `muTopOpt/bounds.py` |
| Optimizers and inner-tolerance control | `muTopOpt/optimize.py` |

References:
* J-FFT preconditioner: https://doi.org/10.1016/j.cma.2026.119269
* Topology optimization: https://arxiv.org/abs/2107.04123

## Install

Requires `muGrid` (≥ 1.4.0, with FFT support), `NuMPI`, `muTimer` and
`numpy`; these are installed as dependencies. Optional extras: `scipy` (for
the serial `optimize_lbfgs`), `mpi4py` (MPI runs) and `pytest` (tests). For
GPU support, `muGrid` must be compiled with CUDA or ROCm/HIP support and you
additionally need `cupy`.

```bash
pip install -e ".[test,scipy]"      # add ,mpi for mpi4py
```

### GPU memory

With `compute.device` set to `"gpu"`, muTopOpt allocates everything on the GPU either as plain
device memory (`hipMalloc`/`cudaMalloc`) or as managed memory
(`hipMallocManaged`/`cudaMallocManaged`), and picks one automatically:

| GPU | ranks | memory | why |
|---|---|---|---|
| discrete (NVIDIA, AMD MI200/MI300X, …) | any | plain | host and device memories are separate, so managed pages migrate between them whenever the other side touches them |
| unified-memory APU (AMD MI300A) | 1 | managed | host and device share one physical memory, so nothing migrates; plain allocations are limited to the ~62.8 GiB device window, managed ones reach all 128 GB |
| unified-memory APU (AMD MI300A) | > 1 | plain | MPI exchanges of managed buffers are much slower than of plain device buffers |

The measurements behind this choice are in the notes above
`_device_has_unified_memory` in `muTopOpt/homogenization.py`.

The run prints its choice and the reason (`device memory: ...`) and records it
in the output file (`device_memory` attribute). `MUTOPOPT_MANAGED=1` or `0`
overrides it. The one case that needs this: a multi-rank run on MI300A whose
fields on one rank exceed the plain window. Either use more ranks or set
`MUTOPOPT_MANAGED=1` and accept the slower exchanges.

With plain memory on MI300A, muTopOpt also sets `MUGRID_UNIFIED_MEMORY=0`,
and on Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1` also
`MUGRID_GPU_AWARE_MPI=1`, unless they are already set. muGrid otherwise takes
plain MI300A memory for host-readable, which it is not.

## Usage

### Python API

```python
from muTopOpt import (SimpMaterial, Homogenization, StressTargetProblem,
                      PhaseFieldRegularization)
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases
from muTopOpt.optimize import initial_density, optimize_lbfgs

material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-3)
homog    = Homogenization((64, 64), material)              # or (n, n, n) for 3D
cases    = target_load_cases(2, isotropic_stiffness_tensor(2, K=0.1, G=0.05))
reg      = PhaseFieldRegularization(homog, weight=1.0)  # weight per unit relative interfacial area
problem  = StressTargetProblem(homog, cases, regularization=reg)

rho0     = initial_density(homog.nb_pixels, kind="random", seed=0)
rho, info = optimize_lbfgs(problem, rho0, maxiter=200)
```

### Command line

`simulate.py` (elasticity) reads a JSON control file. Write one with every key
at its default, edit it, and run it:

```bash
python simulate.py --write-defaults run.json   # also writes simulate.schema.json
python simulate.py run.json
mpirun -np 4 python simulate.py run.json
python simulate.py --help                      # every key, its default and meaning
```

A control file only needs the keys that differ from the defaults; `null`
means the value is determined automatically, as each key's help describes.
Unknown keys are rejected. For example, a 96³ run that restarts from a
previous output (the density is Fourier-interpolated when the grids differ):

```json
{
  "grid": {"nb_grid_pts": [96, 96, 96]},
  "target": {"bulk_modulus": 0.1, "shear_modulus": 0.05},
  "regularization": {"interface_width": 0.02},
  "initial_density": {"source": "cell.nc"},
  "output": {"file": "cell96.nc"}
}
```

`regularization.interface_width` is in the length units of the cell (unit
length by default), so 0.02 is about two grid spacings at 96 points. The
editor can check a file against `simulate.schema.json` (its `"$schema"`
key points there). Every output file stores the complete control it was run
with in its `control` attribute, so a run can be reproduced from its output.

The conduction driver `simulate_conduction.py` keeps its command-line
options (`python simulate_conduction.py --help`).

### Reading the log

`f` is the full objective — the normalized stress misfit summed over all load
cases plus the regularization — reported as the adjoint-corrected Lagrangian
$L = f + \sum_\Gamma v_\Gamma^T (K u_\Gamma - b_\Gamma)$ (with adjoints
$v_\Gamma$; see `muTopOpt/problem.py`) evaluated with the *current* inner CG
tolerance (the `cg-rtol` column). Its evaluation error is
$O(\text{cg-rtol}^2)$, so values printed at *different* `cg-rtol` are not
directly comparable: when the accuracy control tightens the tolerance (typically right
after the first iteration), `f` can appear to jump up by roughly the old
tolerance's evaluation error even though the true objective did not increase.
At constant `cg-rtol` the printed sequence decreases monotonically. Only
*accepted* iterates print a line; the `iters=` count in the final `done:`
line also includes rejected trial steps, so it can exceed the number of
printed iterations.

A `cg-stalled=N` entry in an iterate line counts the inner solves of that
outer step that hit their finite-precision residual floor before the
requested tolerance (common in single precision, where the achievable
relative residual is ~$10^{-6}$ for a healthy right-hand side and far coarser
for the near-round-off right-hand sides that Hessian-vector products can
produce). Such a solve is cut short — on stagnation, divergence, or the
iteration cap — and its best iterate is used (never worse than the zero
solution, with the true residual $b - Kx$ recomputed) instead of aborting
the run; the adjoint-corrected objective and the trust-region accuracy
control account for the extra solve error. `inner_solver.print_iterations`
prints a detail line per cut-short solve.

## License

MIT — see [LICENSE](LICENSE).
