# muTopOpt

FFT-accelerated finite-element **topology optimization** of mechanical
metamaterials, built on [muGrid](https://github.com/muSpectre/muGrid).

muTopOpt designs a periodic unit cell whose homogenized (effective) stiffness
matches a prescribed target — e.g. a given bulk/shear modulus or a negative
Poisson's ratio. It uses an **element-wise (per-pixel) density** with SIMP
material interpolation, a stress-matching objective with phase-field
regularization (no explicit volume constraint), and the **discrete adjoint
method** for exact sensitivities. The micromechanical equilibrium and the
adjoint problem — the two expensive parts of every optimization step — are
solved by muGrid's matrix-free, GPU-capable, J-FFT (Green–Jacobi) preconditioned
conjugate gradient. The method is that of Jödicke et al., *Topology optimization
of metamaterials with FFT-accelerated micromechanical solvers*.

**Dimension-agnostic:** the same code runs 2D and 3D unit cells (a 2D cell needs
3 independent load cases to constrain the effective stiffness, a 3D cell 6).

## Install

Requires an installed `muGrid` (≥ 0.110) with FFT support, plus numpy and scipy.

```bash
pip install -e ".[test]"
```

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
reg      = PhaseFieldRegularization(homog, eta=1.0, well_weight=1e-3)
problem  = StressTargetProblem(homog, cases, regularization=reg)

rho0     = initial_density(homog.nb_pixels, kind="random", seed=0)
rho, info = optimize_lbfgs(problem, rho0, maxiter=200)
```

### Command line

```bash
python simulate.py -n 64 64        --target-K 0.1  --target-G 0.05 --bfgs-maxiter 200
python simulate.py -n 96 96 96     --target-K 0.1  --target-G 0.05 --eta 2.0 --output cell.nc
```

## Design

| Piece | Where |
|-------|-------|
| SIMP interpolation `λ(ρ), μ(ρ)` + derivatives | `material.py` |
| Forward/adjoint solves, homogenized stress, J-FFT precond | `homogenization.py` |
| Stress-matching objective + adjoint sensitivity assembly | `problem.py` |
| Phase-field regularization — element-wise (FD Laplacian) and **nodal FE** (fused scalar FE-Laplacian, P1/Q1) | `regularization.py` |
| Load-case / target-stiffness helpers | `loadcases.py` |
| Optimizer drivers + density initialization | `optimize.py` |

The design variable `ρ` is one value per pixel (element), which maps directly to
muGrid's per-pixel `IsotropicStiffnessOperator`; the adjoint sensitivity is the
fused `compute_sensitivity` kernel followed by the SIMP chain rule, so no full
stiffness tensor or strain/stress field is ever stored. Everything runs on
muGrid fields, so with a GPU build of muGrid the solve and sensitivity are
device-resident.

The outer optimizer is **NuMPI's** bound-constrained, MPI-distributed L-BFGS
(`l_bfgs_bounded`): the density stays in `[0, 1]` by projection, and every
reduction is done over the same domain decomposition as the muGrid fields, so
the optimizer runs correctly on one rank or many. (A serial SciPy L-BFGS-B path,
`optimize_lbfgs`, is kept as a dependency-light alternative.)

## Status

Working and tested (see `test/`):

- Exact adjoint gradient — validated against finite differences in **2D and 3D**,
  with and without regularization, in serial and under MPI.
- End-to-end 2D and 3D optimization reducing the objective toward the target,
  **serial and MPI-parallel** (NuMPI bounded L-BFGS).
- **Nodal FE phase-field** variant (`--density nodal`) with the element-consistent,
  memory-lean fused FE-Laplacian regularization — gradient FD-validated and the
  FE energy identity checked for P1 and Q1 in 2D and 3D.

Roadmap:

- **GPU end-to-end** — exercise the device path on an MI300A (fields already live
  on muGrid collections; add a `--device gpu` wiring).
- **FE-consistent double-well** (Phase 2) — a muGrid shape-function-value (N)
  interpolation operator for a quadrature-consistent double-well (currently
  lumped nodal); optionally a corner-average material map.
- **Decomposition-invariant random init** — `initial_density(kind="random")`
  currently draws per-rank noise, so MPI and serial runs start from different
  fields (the `uniform` init and the gradient are decomposition-invariant).
- Validation against published 2D auxetic designs.

## License

MIT — see [LICENSE](LICENSE).
