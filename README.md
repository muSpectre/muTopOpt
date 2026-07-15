# muTopOpt

FFT-accelerated finite-element **topology optimization** of mechanical
metamaterials, built on [muGrid](https://github.com/muSpectre/muGrid).

muTopOpt designs a periodic unit cell whose homogenized (effective) stiffness
matches a prescribed target — e.g. a given bulk/shear modulus or a negative
Poisson's ratio. It uses a SIMP material interpolation, a stress-matching
objective with phase-field regularization (no explicit volume constraint),
and the discrete adjoint method for exact sensitivities. The micromechanical
equilibrium and the adjoint problem — the two expensive parts of every
optimization step — are solved by muGrid's matrix-free, GPU-capable, J-FFT
(Green–Jacobi) preconditioned conjugate gradient.

See
* J-FFT: https://arxiv.org/abs/2508.02613
* Topology optimization: https://arxiv.org/abs/2107.04123

## Install

Requires an installed `muGrid` with FFT support, plus `numpy` and `scipy`.
For GPU support, `muGrid` must be compiled with CUDA or ROCm/HIP support
and you additionally need `cupy`.

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
# restart from a previous run's output; a coarse design seeds a finer grid
# (the density is Fourier-interpolated when the grids differ)
python simulate.py -n 128 128 128  --target-K 0.1  --target-G 0.05 --init cell.nc
```

### Reading the log

`f` is the full objective — the normalized stress misfit summed over all load
cases plus the regularization — but it is reported as the adjoint-corrected
Lagrangian `L = f + Σ_Γ λ_Γᵀ(K u_Γ − b_Γ)` evaluated with the *current* inner
CG tolerance (the `cg-rtol` column). The correction cancels the first-order
effect of the truncated state solves, leaving an `O(cg-rtol²)` evaluation
error, so values printed at *different* `cg-rtol` are not directly
comparable: when the accuracy control tightens the tolerance (typically right
after the first iteration), `f` can appear to jump up by roughly the old
tolerance's evaluation error even though the true objective did not increase.
At constant `cg-rtol` the printed sequence decreases monotonically. Only
*accepted* iterates print a line; the `iters=` count in the final `done:`
line also includes rejected trial steps, so it can exceed the number of
printed iterations.

A `cg-stalled=N` entry in an iterate line counts the inner solves of that
outer step that hit their finite-precision residual floor before the
requested tolerance (common in `--precision single`, where the achievable
relative residual is ~1e-6 for a healthy right-hand side and far coarser
for the near-round-off right-hand sides that Hessian-vector products can
produce). Such a solve is cut short — on stagnation, divergence, or the
iteration cap — and its best iterate is used (never worse than the zero
solution, with the true residual `b − Kx` recomputed) instead of aborting
the run; the adjoint-corrected objective and the trust-region accuracy
control account for the extra solve error. `--output-cg-iters` prints a
detail line per cut-short solve.

## License

MIT — see [LICENSE](LICENSE).
