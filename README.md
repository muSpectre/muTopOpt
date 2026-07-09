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
```

## License

MIT — see [LICENSE](LICENSE).
