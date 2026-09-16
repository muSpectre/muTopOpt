Change log for muTopOpt
=======================

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
A 512³ design in single precision fits in about 55 GiB on one GPU and runs
entirely on device.

Requirements

`muGrid` provides the FFT engine, the domain decomposition, the fused operators
and the preconditioners. This release needs a muGrid that provides
`NodalMomentOperator` (see the pin in `pyproject.toml`).
