Change log for muTopOpt
=======================

unreleased
----------

- ENH: `--preconditioner hybrid` and `hybrid-jacobi` invert the reference
  stiffness with muGrid's `HybridFourierTridiagonalPreconditioner` (FFT in the
  rank-local axes, a tridiagonal solve in the distributed one) instead of a
  distributed FFT, alone or inside the same J-FFT Jacobi scaling. Same
  operator, same CG iterations; on two GPUs 3-11% faster per CG iteration than
  `green-jacobi` from 128^3 in float32, slower on one GPU
- ENH: `benchmarks/solve_bench.py` runs under `mpirun` (it used to build a
  serial communicator on every rank) and records `nb_ranks` in its CSV
- FIX: `Homogenization` makes the rank's GPU cupy's current device. cupy
  otherwise defaults to device 0, so with one GPU per rank every cupy
  temporary of rank 1 landed on GPU 0 next to fields on GPU 1
- FIX: The inner CG's norms come from muGrid's `linalg.norm_sq` / `vecdot` /
  `axpy_norm_sq` instead of `xp.dot` on the raw field buffer. BLAS `sdot`
  accumulates a float32 field in float32, so its error grows *linearly* in the
  number of entries -- measured 4.2e-9 at 32^3 but 5.5e-6 at 128^3 and ~4.6e-4
  at 512^3, the only quantity in the solve that degrades with resolution this
  way. It also silently materialised a full contiguous copy of the field
  (~`dim * N * 4` bytes, on the device for GPU runs), because `Field.p` is a
  strided view whenever the collection carries ghosts. muGrid's reductions
  accumulate and return in double, skip the ghosts, and copy nothing -- which
  additionally makes `b_norm` bit-identical to the `||b||` the CG itself
  converges against, instead of dividing a double-accumulated residual by a
  float32-accumulated norm
- FIX: The consistent-objective correction `-λᵀr` likewise uses
  `linalg.vecdot` rather than summing a full-size float32 product array. This
  value is the reported objective *and* feeds the trust region's accuracy
  control, so it carries the tightest error budget in the package
- FIX: Inner-CG tolerances are clamped to what the field precision can reach
  (1e-6 in float32, where eps is 1.19e-7 and the true residual `b - Kx`
  stagnates while only the recursive residual keeps shrinking). The floor is
  applied at every point a tolerance is *consumed* -- `solve_rhs`, the adaptive
  controller's construction, and both drivers -- rather than at one branch of
  one driver, which is how it came to be skipped by `simulate.py`'s
  "adaptation disabled" path, by `--cg-tol-min` given explicitly, by
  `simulate_conduction.py` entirely, and by the library defaults `cg_tol=1e-8`
  and `cg_tol_min=1e-10`, none of which float32 can meet.
  `AdaptiveInnerTolerance` stays a pure, dtype-agnostic controller
- ENH: `Homogenization` / `HomogenizationConductivity` take `cg_tol=None`
  (the new default), resolving to a tolerance the chosen precision can reach.
  An explicit unreachable tolerance is raised to the floor with a warning; a
  default the caller never chose is adjusted silently
- ENH: `simulate_conduction.py` gained the precision-aware accuracy limits
  `simulate.py` already had: a single-precision `--bfgs-gtol` default and the
  inner-tolerance floor

v1.0.0 (16Sep26)
----------------

First release. muTopOpt does FFT-accelerated finite-element topology
optimization of periodic metamaterials: it designs a unit cell whose
homogenized stiffness (or conductivity) matches a prescribed target, by the
method of Jödicke et al., *Topology optimization of metamaterials with
FFT-accelerated micromechanical solvers*.

What it does

- Stress-matching objective against a target effective stiffness, given either
  as bulk and shear moduli or as Young's modulus and Poisson's ratio, with
  phase-field regularization and no explicit volume constraint. A conductivity
  analogue (`FluxTargetProblem`) shares the same machinery
- Exact sensitivities by the discrete adjoint method. The finite-difference
  gradient check in `test/test_gradient.py` is the correctness gate for the
  whole pipeline and runs serially and under MPI
- Dimension-agnostic: the same code paths run 2D (3 load cases) and 3D (6)
- Two density discretizations: element-wise (per-pixel, FD-Laplacian penalty)
  and nodal finite-element (element-consistent H¹ seminorm), the latter acting
  as an implicit sensitivity filter so the optimizer can merge or dissolve
  features instead of locking in the initial topology
- Two outer optimizers, both MPI-distributed through NuMPI: a bound-constrained
  L-BFGS and a trust-region Newton-CG with exact Hessian-vector products from
  the second-order adjoint. The trust region is the default where available,
  since its acceptance test compares against a computable predicted reduction
  and so cannot drown in inner-solve noise the way a line search does
- Adaptive inner CG tolerance coupled to the outer optimizer, and
  precision-aware tolerance defaults
- Restart from a previous run's output, Fourier-resampled if the grids differ
- NetCDF output, flushed per frame, carrying the full invocation and the
  versions that produced it

Scale

No stiffness tensor and no strain or stress field is ever stored: the operator,
the preconditioner and the sensitivity are all matrix-free and fused, and all
fields share one ghosted, MPI-decomposed, optionally device-resident layout.
A 512³ design in single precision fits on a single 128 GB unified-memory
accelerator -- measured above 60 GiB resident during the first L-BFGS
iterations of an MI300A run -- and runs entirely on device.

Requirements

`muGrid` provides the FFT engine, the domain decomposition, the fused operators
and the preconditioners. This release needs a muGrid that provides
`NodalMomentOperator` (see the pin in `pyproject.toml`).
