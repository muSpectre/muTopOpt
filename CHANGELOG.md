<!--
Style for this changelog (keep to it so entries stay uniform):

- Newest version first. Heading `vX.Y.Z (DDMonYY)`, underlined with dashes.
  A version not yet released is headed `vX.Y.Z (not yet released)`.
- One bullet per user-visible change, at most two lines (about 200
  characters). Say what changed for the user and, if useful, why or by how
  much. Omit internals: no function, class, file or variable names, no
  environment variables, no commit or PR references.
- Start every bullet with one of: `ENH:` (new feature), `BUG:` (fix),
  `API:` (interface change), `PERF:` (speed or memory), `MAINT:`, `TST:`,
  `DOC:`, `BUILD:`, `CI:`.
- Small related fixes may share one bullet. No trailing period on bullets.
-->

Change log for muTopOpt
=======================

v1.2.0 (18Oct26)
----------------

- BUG: The element-wise phase-field regularization mis-scaled its gradient penalty
  on anisotropic voxels; with muGrid 1.5.0 or newer it uses the spacing of every axis
- PERF: Managed GPU memory is used only on a single-rank unified-memory accelerator.
  Multi-rank runs use plain device memory, since MPI is 2-5x slower on managed buffers
- ENH: The memory choice can still be overridden; it is printed at startup together
  with its reason and recorded in the output file
- ENH: With plain memory on a unified-memory accelerator, muGrid's host-staging and
  GPU-aware-MPI switches are set automatically unless already set

v1.1.0 (05Oct26)
----------------

- ENH: The output file records the elastic response per frame: homogenized stresses
  per load case and the isotropic-equivalent moduli, plus the targets and histories
- BUG: Run-state attributes and histories are checkpointed with every dumped frame,
  so a run killed at the walltime leaves a readable file instead of placeholders
- BUG: The final frame and summary use the last accepted iterate, not a rejected
  trust-region trial
- API: The regularization weight is now the penalty per unit relative interfacial
  area, independent of cell size and dimension (earlier weights: multiply by one third)
- BUG: Every rank count starts from the same initial design; a 2-rank run used to
  start from two copies of a half-size design
- ENH: The hybrid FFT/tridiagonal preconditioner from muGrid is available, alone or
  with Jacobi scaling; 3-11% faster per iteration on two GPUs at 128³ and above
- ENH: The solve benchmark runs under MPI and records the rank count
- BUG: Each rank's GPU is made the current device, so temporaries no longer land on
  GPU 0 next to fields on another GPU
- BUG: The inner solver's norms and the objective correction are accumulated in
  double without copying the field; in float32 their error used to grow with grid size
- BUG: Inner-solve tolerances are clamped to what the precision can reach wherever
  they are consumed, so float32 runs no longer burn the iteration budget
- ENH: The default inner tolerance follows the chosen precision; an unreachable
  explicit one is raised with a warning. The conduction driver gained the same limits

v1.0.0 (16Sep26)
----------------

- ENH: First release: FFT-accelerated finite-element topology optimization of
  periodic metamaterials whose homogenized stiffness or conductivity matches a target
- ENH: Stress-matching objective with phase-field regularization and exact
  sensitivities by the discrete adjoint method, in 2D and 3D, serially and under MPI
- ENH: Element-wise and nodal density discretizations, the latter acting as an
  implicit sensitivity filter
- ENH: MPI-distributed bound-constrained L-BFGS and trust-region Newton-CG optimizers
  with exact Hessian-vector products; adaptive, precision-aware inner tolerances
- ENH: Restart from a previous run's output with Fourier resampling; NetCDF output
  flushed per frame with the full invocation and versions
- PERF: Matrix-free and fused throughout: a 512³ design in single precision fits and
  runs on one 128 GB unified-memory accelerator
