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

v1.5.0 (not yet released)
-------------------------

- PERF: Plain device memory is used on every GPU, also for a single rank on a
  unified-memory accelerator, where MPI is ~40x slower on managed buffers
- ENH: Managed memory is still available on request
- PERF: The solves of the trust region's Hessian-vector products adapt their
  tolerance to the subproblem they serve, between 1e-2 early and 1e-4 near
  convergence, instead of a fixed 1e-3 that cost twice the CG iterations early on
- PERF: Hessian solves without warm start skip the operator apply that checks a warm
  start, and the CG fallback snapshot is only refreshed when the residual has halved
- API: `optimizer.trust_region.hessian_tolerance` defaults to null (adaptive); a
  number still fixes it
- PERF: The element-wise sensitivity work of the gradient and of the Hessian-vector
  products runs on the GPU, in double precision, instead of in NumPy on the host;
  only the gradient and the product are copied to the host
- PERF: The phase-field regularizations, including the consistent double well, also
  compute on the GPU; their results come back as host or device arrays like their input
- TST: GPU tests compare gradient, Hessian product and regularization with the CPU;
  they are skipped without CuPy and a GPU build of muGrid
- PERF: The trust region iterates on the GPU: design, gradient and Hessian products
  stay in device memory, and only single numbers come to the host
- BUILD: Requires NuMPI 0.19.0 or newer

v1.4.0 (28Oct26)
----------------

- ENH: Control files can also be written in TOML; a file whose name ends in
  `.toml` is read as TOML, anything else as JSON
- BUG: The restart control file numbers the next output file with a `.restart<N>`
  suffix; 1.3.0 incremented any trailing number, such as a parameter in the name
- BUILD: Requires tomli on Python 3.10

v1.3.0 (27Oct26)
----------------

- ENH: Installing the package provides the `mutopopt` command, which runs the
  elasticity driver; `python simulate.py` from the repository keeps working
- ENH: Each rank picks its GPU from its rank on the node as MPI reports it, so GPU
  placement works with any MPI library and launcher
- MAINT: No settings specific to an MPI library or machine are made for muGrid any
  more; muGrid detects GPU-aware MPI and host-readable memory itself
- ENH: The log and the NetCDF output say whether MPI takes device buffers directly
- API: MPI runs no longer need mpi4py: muGrid starts MPI, and the optimizers reduce
  over the muGrid communicator; an mpi4py communicator is still accepted
- ENH: Optional detailed timing breaks every inner solve down into CG steps, halo
  exchange, stencil and preconditioner phases, synchronized with the GPU if wanted
- ENH: Every run writes a restart control file that continues it from its output file
  into a new, numbered one, so a run can be resumed job after job with one command
- TST: The tests run on the real MPI decomposition under mpirun
- BUILD: Requires muGrid 1.6.0 or newer and a NuMPI release newer than 0.15.2

v1.2.1 (18Oct26)
----------------

- BUILD: The package version is taken from the git tag; the 1.2.0 source archive
  wrongly reported itself as 1.1.0

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
