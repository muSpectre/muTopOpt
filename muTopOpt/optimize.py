#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Gradient-based outer optimizers for the density design problem.

Every driver minimizes ``f(ρ)`` over the box ``lo ≤ ρ ≤ hi`` (default
``[0, 1]``) for a problem object providing
``objective_and_gradient(rho) -> (f, g)``, with ``rho`` shaped like
:attr:`Homogenization.nb_pixels` (the rank-local slice under MPI).

Stationarity measure
--------------------
With ``Π(x) = clip(x, lo, hi)`` the projection onto the box, the free
(box-KKT) gradient ``g_free`` is ``g`` with every component set to zero where
``ρ_i`` sits on a bound and ``g_i`` points further into it (``ρ_i = lo`` and
``g_i ≥ 0``, or ``ρ_i = hi`` and ``g_i ≤ 0``). All drivers stop when

    ‖g_free‖_∞ ≤ gtol · V_e / V,

i.e. ``gtol`` is stated for the mesh-invariant volume-fraction derivative
``ĝ = (V/V_e) g`` (``V`` cell volume, ``V_e`` pixel volume; see
:func:`_gradient_scale`).

Drivers
-------
:func:`optimize_bounded_lbfgs`
    NuMPI's MPI-distributed projected L-BFGS (``l_bfgs_bounded``). The search
    direction ``d = -H g_free`` comes from the two-loop recursion over the
    last ``maxcor`` curvature pairs ``(s, y) = (ρ_new - ρ, g_free,new -
    g_free)`` (a pair is stored only if ``sᵀy > 1e-12 ‖s‖ ‖y‖``). The step
    ``ρ_new = Π(ρ + α d)`` is found by halving ``α`` (from 1) along the
    projection arc until the Armijo condition
    ``f(ρ_new) ≤ f(ρ) + c₁ α g_freeᵀ d`` holds.
:func:`optimize_trust_region`
    NuMPI's MPI-distributed bound-constrained trust-region Newton-CG
    (``tr_newton_bounded``), using exact Hessian-vector products ``H v`` from
    the problem's second-order adjoint. Each iteration approximately
    minimizes the model ``m(s) = g_freeᵀ s + ½ sᵀ H s`` subject to
    ``‖s‖₂ ≤ Δ`` on the free subspace with Steihaug--Toint truncated CG
    (stop on the CG residual tolerance, or step to the trust-region boundary
    on negative curvature or when the CG iterate would leave it). The trial
    point ``ρ_try = Π(ρ + s)`` is judged by the ratio of actual to predicted
    reduction,

        r = (f(ρ) - f(ρ_try)) / pred,   pred = m(0) - m(ρ_try - ρ),

    accepted if ``r ≥ eta``, and the radius is updated as ``Δ ← Δ/4`` if
    ``r < 1/4`` and ``Δ ← min(2Δ, Δ_max)`` if ``r > 3/4`` and the step
    reached the boundary (unchanged otherwise).
:func:`optimize_lbfgs`
    SciPy's serial L-BFGS-B, for single-process runs without NuMPI.

Choice of optimizer: the trust-region method needs a problem with Hessian
support (``StressTargetProblem(..., hessian=True)``) and costs two extra CG
solves per load case per Hessian-vector product, but its acceptance test
compares against the computable ``pred`` and can require the objective
evaluation error to stay below ``eta_f · pred``; an Armijo line search
instead has to resolve decreases proportional to ``α``, which fall below the
objective noise of truncated inner CG solves as ``α → 0``. ``simulate.py``
therefore selects the trust region whenever the installed NuMPI provides it,
and projected L-BFGS otherwise; ``simulate_conduction.py`` uses projected
L-BFGS because :class:`~muTopOpt.conduction_problem.FluxTargetProblem` has
no Hessian-vector product.

Both NuMPI drivers take ``x``/gradient/bounds as rank-local slices and the
scalar objective as a global reduction -- the MPI contract of the problems'
``objective_and_gradient``.
"""

import numpy as np

from .precision import cg_rtol_floor


def _gradient_scale(problem):
    """``V_e/V``: converts a *mesh-invariant* ``gtol`` -- measured on the
    volume-fraction derivative ``ĝ = (V/V_e) ∂f/∂ρ``, i.e. the derivative of
    the objective with respect to a density perturbation occupying a unit
    *fraction* of the cell -- to the raw per-variable gradient the inner
    optimizer (NuMPI/SciPy) compares against.

    The objective matches *volume-averaged* stresses, so a single design
    variable's gradient entry is ``O(V_e/V) = O(1/N)``: an absolute per-pixel
    ``gtol`` would mean a different physical stationarity at every resolution
    (and would certify the unchanged initial design as converged on
    sufficiently fine grids). Both the element and the nodal design have one
    variable per pixel of the periodic grid, so one scale covers both.
    MPI-safe: ``grid_spacing`` and ``domain_volume`` are global quantities,
    identical on every rank."""
    h = getattr(problem, "h", None)
    if h is None:
        return 1.0
    return float(np.prod(h.grid_spacing)) / h.domain_volume  # = 1/N_global


class AdaptiveInnerTolerance:
    """Eisenstat--Walker-style *forcing term* coupling the inner CG tolerance
    to the outer optimizer's stationarity, with a stagnation ratchet.

    The inner (forward/adjoint) CG relative tolerance starts coarse at
    ``rtol_start`` and is retuned once per accepted outer iterate. It is
    **monotone non-increasing** (never loosens) and floored at ``rtol_min``.
    With ``γ_k = ‖g_free,k‖_∞`` the gradient norm last passed to
    :meth:`observe` (in mesh-invariant ``ĝ`` units), ``γ_0`` the first one
    and ``γ_best`` the smallest one that counted as progress, the update
    applied by :meth:`advance` at accepted iterate ``k`` is

        if γ_k < (1 - stall_rel) γ_best:   (progress; γ_best ← γ_k)
            t_k = clip(c · rtol_start · (γ_k / γ_0)^alpha, rtol_min, rtol_start)
        else:                              (stagnation)
            t_k = shrink · rtol_k
        rtol_{k+1} = max(rtol_min, min(rtol_k, t_k))

    with ``rtol_0 = rtol_start`` (``t_k = rtol_min`` if ``γ_0 = 0``). Two
    mechanisms thus drive the tolerance down:

    * **Forcing term.** On progress the tolerance tracks the relative forcing
      term (Eisenstat--Walker "choice 2" form): coarse while far from a
      stationary point (no oversolving), tightening in proportion to the
      gradient as it approaches one. ``γ`` is the box-KKT gradient norm, the
      same stationarity measure ``l_bfgs_bounded`` converges on (NuMPI
      ``_kkt_residual``). The *relative* form ``γ_k / γ_0`` makes the
      coupling independent of the units of ``γ`` and hence of the mesh
      resolution; an absolute term ``c γ_k^alpha`` would compare a
      mesh-dependent quantity against the mesh-independent tolerances.

    * **Stagnation ratchet.** If ``γ_k`` fails to beat ``γ_best`` by the
      relative margin ``stall_rel``, the outer optimizer has stalled at the
      current accuracy and the tolerance is multiplied by ``shrink``. The
      forcing term alone cannot do this: it only tightens when ``γ``
      decreases, but noise from loose inner solves can stop ``γ`` from
      decreasing, which would hold the tolerance at its current value
      indefinitely. Refining the inner accuracy on stagnation is the
      mechanism of inexact trust-region PDE-constrained optimization
      (Heinkenschloss & Vicente 2001; Kouri, Heinkenschloss, Ridzal & van
      Bloemen Waanders 2013/14).

    **Frozen between iterates.** :meth:`advance` retunes the tolerance once per
    accepted outer iterate (called from the optimizer callback); within an
    iterate's line search :attr:`current` is constant, so every trial point is
    solved to the *same* accuracy. That -- together with monotonicity -- keeps
    the L-BFGS secant pair ``y = g_new - g_old`` consistent (both gradients
    carry the same solve accuracy), the condition under which quasi-Newton
    methods tolerate inexact gradients (Xie, Byrd & Nocedal 2020; Berahas, Byrd
    & Nocedal 2019).

    Mathematical basis: Dembo--Eisenstat--Steihaug (1982) and Eisenstat--Walker
    (1996) for the forcing term; Carter (1991) / Byrd--Chin--Nocedal--Wu (2012)
    for the gradient norm condition ``||g_approx - g|| <= eta ||g_approx||``.

    Notes
    -----
    The floor ``rtol_min`` matters: :class:`~muTopOpt.problem.StressTargetProblem`
    with ``consistent_objective=True`` reports an objective that is *second*
    order in the CG residual, but the returned gradient is only *first* order,
    so the final gradient error is governed by ``rtol_min``. Pick it small
    enough that the gradient error falls below the outer ``gtol`` -- otherwise
    L-BFGS cannot certify convergence. In the worst case (persistent
    stagnation) the ratchet drives ``current`` all the way to ``rtol_min``, so
    the run degrades to a fixed-``rtol_min`` solve rather than getting stuck.

    Parameters
    ----------
    rtol_start : float
        Initial (coarsest) inner relative tolerance; also the upper clip of
        the forcing term.
    rtol_min : float
        Floor of the tolerance.
    c, alpha : float, optional
        Coefficient and exponent of the forcing term (defaults 1.0, 1.0).
    stall_rel : float, optional
        Minimum relative decrease of ``γ`` counted as progress (default 0.01).
    shrink : float, optional
        Factor applied to the tolerance on stagnation (default 0.3).
    bounds : tuple, optional
        Box ``(lo, hi)`` of the design variable (default ``(0.0, 1.0)``);
        read by the problems to mask the gradient to ``g_free`` before
        calling :meth:`observe`.
    """

    def __init__(self, rtol_start, rtol_min, c=1.0, alpha=1.0, stall_rel=1e-2,
                 shrink=0.3, bounds=(0.0, 1.0)):
        self.rtol_start = float(rtol_start)
        self.rtol_min = float(rtol_min)
        self.c = float(c)
        self.alpha = float(alpha)
        self.stall_rel = float(stall_rel)
        self.shrink = float(shrink)
        self.bounds = bounds
        self.current = float(rtol_start)
        self._latest_gnorm = None
        # First observed ||g_free||: the reference of the relative forcing
        # term. Fixed by the first observe(), i.e. the initial iterate.
        self._g_first = None
        # Smallest ||g_free|| seen so far; a new gradient must beat this (by
        # stall_rel) to count as progress. +inf so the first iterate always
        # registers as progress.
        self._best_gnorm = float("inf")
        # (gnorm, rtol, stalled) at each advance, for diagnostics / output.
        self.history = []

    def observe(self, gnorm):
        """Record the box-KKT gradient norm of the most recent evaluation.

        Called on *every* objective/gradient evaluation (including line-search
        trials); it only stores the value -- :attr:`current` is untouched until
        :meth:`advance`."""
        self._latest_gnorm = float(gnorm)
        if self._g_first is None:
            self._g_first = self._latest_gnorm

    @property
    def latest_gnorm(self):
        """Box-KKT gradient norm of the most recent evaluation (mesh-invariant
        ĝ units), or ``None`` before the first :meth:`observe`."""
        return self._latest_gnorm

    def advance(self):
        """Retune :attr:`current` from the last observed gradient norm.

        Invoked once per accepted outer iterate (from the optimizer callback).
        Monotone non-increasing and floored at ``rtol_min``. A no-op until the
        first :meth:`observe`."""
        if self._latest_gnorm is None:
            return self.current
        g = self._latest_gnorm
        if g < (1.0 - self.stall_rel) * self._best_gnorm:
            # Genuine progress: let the (relative) forcing term set the
            # target -- units of the observed norm cancel in g / g_first.
            self._best_gnorm = g
            if self._g_first > 0.0:
                target = (self.c * self.rtol_start
                          * (g / self._g_first) ** self.alpha)
            else:
                target = self.rtol_min
            target = min(self.rtol_start, max(self.rtol_min, target))
            stalled = False
        else:
            # Stagnation at the current accuracy: refine (ratchet down).
            target = self.current * self.shrink
            stalled = True
        # Monotone: only ever tighten, never below the floor.
        self.current = max(self.rtol_min, min(self.current, target))
        self.history.append((g, self.current, stalled))
        return self.current


def _problem_rtol_floor(problem):
    """Smallest inner-CG rtol the problem's field precision can reach.

    0.0 for a double-precision problem, and for anything that does not expose
    a dtype, so every clamp below is a no-op there.
    """
    dtype = getattr(getattr(problem, "h", None), "dtype", None)
    return 0.0 if dtype is None else cg_rtol_floor(dtype)


def _make_inner_tolerance(problem, cg_tol_start, cg_tol_min, cg_forcing_c,
                          cg_forcing_exp, cg_stall_rel, cg_stall_shrink,
                          bounds):
    """Build and attach an :class:`AdaptiveInnerTolerance` to ``problem`` when
    adaptive coupling is requested (``cg_tol_start`` given), else detach any
    previous controller and return ``None`` (fixed ``Homogenization.cg_tol``).

    ``cg_tol_min`` defaults to the homogenization's fixed ``cg_tol`` -- i.e.
    "start coarse at ``cg_tol_start`` and tighten down to the tolerance you
    would otherwise have used throughout" -- and is raised to the floor the
    problem's precision can actually reach. That clamp lives here rather than
    in :class:`AdaptiveInnerTolerance`, which stays a pure, dtype-agnostic
    controller."""
    if cg_tol_start is None:
        problem.inner_tolerance = None
        return None
    if cg_tol_min is None:
        cg_tol_min = getattr(problem.h, "cg_tol", 1e-8)
    cg_tol_min = max(cg_tol_min, _problem_rtol_floor(problem))
    cg_tol_start = max(cg_tol_start, cg_tol_min)
    controller = AdaptiveInnerTolerance(
        cg_tol_start, cg_tol_min, c=cg_forcing_c, alpha=cg_forcing_exp,
        stall_rel=cg_stall_rel, shrink=cg_stall_shrink, bounds=bounds,
    )
    problem.inner_tolerance = controller
    return controller


def initial_density(shape, kind="uniform", volume_fraction=0.5, seed=0,
                    smoothing=2, length=None, grid_spacing=None, contrast=0.5,
                    subdomain_locations=None, nb_subdomain_grid_pts=None):
    """Build an initial element-wise density.

    ``shape`` is the *global* grid. Under MPI pass the rank's
    ``subdomain_locations`` and ``nb_subdomain_grid_pts`` (e.g.
    ``homog.engine.subdomain_locations`` and ``homog.nb_pixels``): the field is
    then generated on the whole grid, identically on every rank, and only this
    rank's slice is returned -- so a parallel run starts from exactly the design
    a serial run does. Passing the *local* shape as ``shape`` instead would draw
    and filter an independent field per subdomain, making the starting design
    depend on the number of ranks.

    ``kind='uniform'`` fills with ``volume_fraction``.

    ``kind='random'`` draws uniform noise, applies ``smoothing`` periodic
    three-point box-filter sweeps per axis, standardizes the result to ``z``
    (zero mean, unit standard deviation) and returns
    ``clip(0.5 + 0.5 z, 0, 1)`` -- a cheap low-pass start with no controlled
    length scale. ``volume_fraction`` and ``contrast`` are not used.

    ``kind='filtered_random'`` draws white noise and applies a **periodic
    Gaussian filter** of correlation length ``length`` (in the *physical* units
    of the unit cell -- the cell has length 1 by default, converted to pixels
    via ``grid_spacing``, which defaults to ``1/n`` per axis). The field is then
    standardized to ``z`` and mapped to
    ``clip(volume_fraction + contrast * z, 0, 1)``, so it has (up to the
    clipping) the requested mean volume fraction with smooth blobs of the
    chosen size and no sharp interfaces; the phase-field regularization
    sharpens them during the optimization (no filter is applied there). The
    filter multiplies the DFT of the noise by
    ``Π_α exp(-2π² σ_α² f_α²)``, ``σ_α = length / h_α`` (in pixels) and
    ``f_α`` the frequency in cycles per pixel.

    Choosing ``length`` -- the initial blob size must be **larger** than the
    phase-field interface width the regularization will impose. In the
    Modica-Mortola normalization that width is simply ``eta`` itself (which
    defaults to one grid spacing, ``eta = h``), so pick ``length ~= 2..4 *
    eta`` -- the regularization then *sharpens the blob boundaries* rather
    than dissolving the blobs.

    Returns
    -------
    np.ndarray
        Float64 density with shape ``shape``, or ``nb_subdomain_grid_pts``
        when the subdomain is given.

    Raises
    ------
    ValueError
        If only one of ``subdomain_locations``/``nb_subdomain_grid_pts`` is
        given, if ``kind='filtered_random'`` has no ``length``, or for an
        unknown ``kind``.
    """
    if (subdomain_locations is None) != (nb_subdomain_grid_pts is None):
        raise ValueError("pass both subdomain_locations and "
                         "nb_subdomain_grid_pts, or neither")
    rho = _global_initial_density(shape, kind, volume_fraction, seed,
                                  smoothing, length, grid_spacing, contrast)
    if subdomain_locations is None:
        return rho
    return rho[tuple(slice(int(lo), int(lo) + int(n)) for lo, n in
                     zip(subdomain_locations, nb_subdomain_grid_pts))].copy()


def _global_initial_density(shape, kind, volume_fraction, seed, smoothing,
                            length, grid_spacing, contrast):
    """The initial density on the full grid ``shape``; see initial_density."""
    if kind == "uniform":
        return np.full(shape, float(volume_fraction))
    if kind == "random":
        rng = np.random.default_rng(seed)
        rho = rng.random(shape)
        # Cheap periodic low-pass: repeated box smoothing along every axis.
        for _ in range(int(smoothing)):
            for ax in range(rho.ndim):
                rho = (
                              rho
                              + np.roll(rho, 1, axis=ax)
                              + np.roll(rho, -1, axis=ax)
                      ) / 3.0
        rho -= rho.mean()
        std = rho.std()
        if std > 0:
            rho /= std
        rho = np.clip(0.5 + 0.5 * rho, 0.0, 1.0)
        return rho
    if kind == "filtered_random":
        if length is None:
            raise ValueError("filtered_random needs a correlation `length`")
        shape = tuple(int(n) for n in shape)
        ndim = len(shape)
        if grid_spacing is None:
            grid_spacing = [1.0 / n for n in shape]
        grid_spacing = [float(h) for h in grid_spacing]

        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(shape)

        # Periodic Gaussian low-pass in Fourier space. Per axis the Gaussian has
        # std sigma_px = length / h pixels; its DFT multiplier is
        # exp(-2 pi^2 sigma_px^2 f^2) with f = fftfreq (cycles/pixel).
        spec = np.fft.fftn(noise)
        for ax in range(ndim):
            sigma_px = float(length) / grid_spacing[ax]
            f = np.fft.fftfreq(shape[ax])
            g = np.exp(-2.0 * np.pi ** 2 * sigma_px ** 2 * f ** 2)
            spec *= g.reshape([-1 if d == ax else 1 for d in range(ndim)])
        rho = np.fft.ifftn(spec).real

        rho -= rho.mean()
        std = rho.std()
        if std > 0:
            rho /= std
        return np.clip(float(volume_fraction) + float(contrast) * rho, 0.0, 1.0)
    raise ValueError(f"unknown initial-density kind '{kind}'")


def optimize_bounded_lbfgs(problem, rho0, comm=None, maxiter=200, gtol=2.5,
                           ftol=0.0, xtol=0.0, bounds=(0.0, 1.0), maxcor=10,
                           callback=None, cg_tol_start=None, cg_tol_min=None,
                           cg_forcing_c=1.0, cg_forcing_exp=1.0,
                           cg_stall_rel=1e-2, cg_stall_shrink=0.3,
                           ):
    """Minimize ``problem`` from ``rho0`` with NuMPI's MPI-distributed,
    box-constrained (projected) L-BFGS; see the module docstring for the
    algorithm.

    Parameters
    ----------
    problem : object
        Provides ``objective_and_gradient(rho) -> (f, g)``, a ``last`` dict of
        diagnostics, and (for the adaptive tolerance) an ``h`` homogenization
        and an ``inner_tolerance`` attribute.
    rho0 : array_like
        Initial density, shape :attr:`Homogenization.nb_pixels`.
    maxiter : int, optional
        Maximum number of outer iterations (default 200).
    gtol : float, optional
        Convergence tolerance on the infinity norm of the box-projected
        gradient, measured on the *mesh-invariant* volume-fraction derivative
        ``ĝ = (V/V_e) ∂f/∂ρ`` (see :func:`_gradient_scale`; typical initial
        designs start at ``||ĝ||_inf ~ O(100)``), so the same value means the
        same physical stationarity at every resolution. ``info['max_grad']``
        is reported in the same units.
    comm : muGrid.Communicator, optional
        The communicator over which the density is distributed (the *same*
        decomposition as the muGrid fields, i.e. ``problem.h.comm``); NuMPI
        reduces over it through its communicator protocol, so mpi4py is not
        needed. An mpi4py communicator works too. ``None`` runs serially.
        Density stays in ``bounds`` by projection.
    ftol : float, optional
        Stop when ``|f_new - f| < ftol · max(1, |f_new|, |f|)``; 0 (default)
        disables the test.
    xtol : float, optional
        Stop when ``max_i |ρ_new,i - ρ_i| < xtol`` (global infinity norm of
        the step); 0 (default) disables the test.
    bounds : tuple of float, optional
        Box ``(lo, hi)`` of the density (default ``(0.0, 1.0)``).
    maxcor : int, optional
        Number of stored L-BFGS curvature pairs (default 10).
    callback : callable, optional
        ``callback(it, rho, problem.last)`` after each accepted iterate, with
        ``it`` the 1-based count of accepted iterates; returning ``True`` is
        passed on to NuMPI's callback hook.
    cg_tol_start : float, optional
        Enable *adaptive* inner CG tolerance: the forward/adjoint solves start
        at this (coarse) relative tolerance and tighten as the outer projected
        gradient shrinks (an Eisenstat--Walker forcing term; see
        :class:`AdaptiveInnerTolerance`). ``None`` (default) uses the fixed
        ``Homogenization.cg_tol`` throughout.
    cg_tol_min : float, optional
        Floor for the adaptive tolerance; defaults to the homogenization's
        ``cg_tol``. Must be small enough that the final gradient error is below
        ``gtol`` (the norm condition), or convergence cannot be certified.
    cg_forcing_c, cg_forcing_exp : float, optional
        Coefficient ``c`` and exponent ``alpha`` in the relative forcing term
        ``rtol = c * rtol_start * (||g_free|| / ||g_0||)**alpha`` (defaults
        1.0, 1.0). ``alpha=1`` gives ``rtol = O(||g||)`` and fast local
        convergence.
    cg_stall_rel, cg_stall_shrink : float, optional
        Stagnation ratchet (see :class:`AdaptiveInnerTolerance`): when the
        projected gradient fails to decrease by at least ``cg_stall_rel``
        (default 0.01) over an outer iterate, the tolerance is forced down by
        ``cg_stall_shrink`` (default 0.3) toward ``cg_tol_min``, so a
        noise-limited plateau cannot stall the run.

    Returns
    -------
    rho_opt : np.ndarray
        Final density, same shape as ``rho0``.
    info : dict
        ``success``, ``message``, ``nit`` (outer iterations), ``objective``,
        ``max_grad`` (final ``‖g_free‖_∞`` in ``ĝ`` units), ``history`` (the
        objective of every accepted iterate) and ``cg_rtol_history`` (the
        :attr:`AdaptiveInnerTolerance.history` list, or ``None`` with a fixed
        tolerance).
    """
    from NuMPI.Optimization import l_bfgs_bounded

    history = []
    inner_tol = _make_inner_tolerance(
        problem, cg_tol_start, cg_tol_min, cg_forcing_c, cg_forcing_exp,
        cg_stall_rel, cg_stall_shrink, bounds)

    # The local density grid is passed as-is: NuMPI's l_bfgs_bounded handles an
    # n-D x0 (keeping its iterate/gradient/history flat internally) and returns
    # the result in the same shape.
    def fun(x):
        return problem.objective_and_gradient(x)

    def _cb(x):
        # A new outer iterate has been accepted: retune the inner tolerance
        # from the (just-evaluated) accepted point before the next iterate's
        # line search, so that line search runs at a single, consistent
        # accuracy.
        if inner_tol is not None:
            inner_tol.advance()
        history.append(problem.last.get("objective"))
        if callback is not None:
            return callback(len(history), x, problem.last)
        return False

    # NuMPI compares the raw per-variable gradient; gtol/max_grad are in
    # mesh-invariant ĝ units (see _gradient_scale).
    gscale = _gradient_scale(problem)
    res = l_bfgs_bounded(
        fun, np.asarray(rho0, dtype=float), jac=None,
        bounds_lo=bounds[0], bounds_hi=bounds[1],
        gtol=gtol * gscale, ftol=ftol, xtol=xtol, maxiter=maxiter,
        maxcor=maxcor, comm=comm, callback=_cb
    )
    info = {
        "success": bool(res.success),
        "message": res.message,
        "nit": int(res.nit),
        "objective": float(res.fun),
        "max_grad": float(res.get("max_grad", np.nan)) / gscale,
        "history": history,
        "cg_rtol_history": (inner_tol.history if inner_tol is not None
                            else None),
    }
    return np.asarray(res.x), info


#: Bounds and factor of the adaptive Hessian-solve tolerance (hessian_rtol)
HESSIAN_RTOL_MAX = 1e-2
HESSIAN_RTOL_MIN = 1e-4
HESSIAN_RTOL_FACTOR = 0.1


def hessian_rtol(steihaug_tol):
    """Relative tolerance of the solves in a Hessian-vector product requested
    by a Steihaug CG that solves to the relative residual `steihaug_tol`.

    A product with relative error ``e`` perturbs the Steihaug residual by
    about ``e`` relative to it, so ``e`` an order of magnitude below the
    Steihaug tolerance keeps the subproblem's own accuracy target meaningful
    while the products stay no more accurate than it can use. The result is
    clipped to [:data:`HESSIAN_RTOL_MIN`, :data:`HESSIAN_RTOL_MAX`]: the
    first subproblems (Steihaug tolerance 0.5) get 1e-2, where a fixed 1e-3
    cost twice the CG iterations, and the tolerance tightens only as the
    outer iteration converges. ``None`` (no tolerance given) gives the
    upper bound.
    """
    if steihaug_tol is None:
        return HESSIAN_RTOL_MAX
    return float(min(HESSIAN_RTOL_MAX,
                     max(HESSIAN_RTOL_MIN,
                         HESSIAN_RTOL_FACTOR * steihaug_tol)))


def optimize_trust_region(problem, rho0, comm=None, maxiter=200, gtol=2.5,
                          bounds=(0.0, 1.0), delta0=0.05, delta_max=0.5,
                          eta=0.1, eta_f=0.25, cg_tol_start=None,
                          cg_tol_min=1e-10, hv_rtol=None, callback=None,
                          disp=False):
    """Minimize ``problem`` from ``rho0`` with NuMPI's MPI-distributed,
    bound-constrained **trust-region Newton-CG** (``tr_newton_bounded``); see
    the module docstring for the Steihaug--Toint subproblem solve, the ratio
    test and the radius update.

    Requires a problem constructed with ``hessian=True`` (the per-load-case
    state caching and second-order-adjoint Hessian-vector product).

    Robustness to truncated inner solves: unlike a line search -- whose
    sufficient-decrease test must resolve decreases that shrink towards zero
    and hence eventually drowns in the CG-induced objective noise -- the
    trust-region acceptance ratio compares against the *computable* predicted
    reduction ``pred``, and the driver enforces the inexact-trust-region
    condition ``|f_err| <= eta_f * pred`` through two hooks wired here:

    * ``fun_error``: ``e = Σ_Γ |v_Γᵀ r_Γ|``, the adjoint-weighted residuals of
      the forward solves (``problem.last['corrections']``) -- a computable
      first-order estimate of the objective evaluation error that the
      consistent (Lagrangian) objective already computes;
    * ``request_accuracy(target)``: with the consistent objective the error
      is second order, ``f_err ≈ C rtol²``, so the state-solve tolerance is
      set to ``rtol ← max(cg_tol_min, min(rtol, ½ rtol √(target / e)))``
      (``rtol ← max(cg_tol_min, rtol / 10)`` if ``e`` or ``target`` is 0).
      The factor ½ is a safety margin.

    Parameters
    ----------
    problem, rho0 : see :func:`optimize_bounded_lbfgs`
        ``problem`` must additionally provide ``hessian`` (true),
        ``ensure_state(rho)``, ``hessian_vector_product(v, rtol)`` and
        ``last_hv_cg_iters``.
    maxiter : int, optional
        Maximum number of outer iterations, counting rejected trial steps
        (default 200).
    gtol : float, optional
        Convergence tolerance on the box-projected gradient, in the same
        mesh-invariant units as :func:`optimize_bounded_lbfgs` (the
        volume-fraction derivative ``ĝ = (V/V_e) ∂f/∂ρ``; default 2.5).
    comm : muGrid.Communicator, optional
        Communicator of the domain decomposition (as for
        :func:`optimize_bounded_lbfgs`).
    bounds : tuple of float, optional
        Box ``(lo, hi)`` of the density (default ``(0.0, 1.0)``).
    delta0, delta_max : float, optional
        Initial/maximum trust-region radius in RMS-per-pixel density units:
        the Euclidean radius is ``Δ = delta · √N`` with ``N`` the global
        number of design variables (defaults 0.05 / 0.5).
    eta : float, optional
        Step acceptance threshold on the ratio ``r = ared/pred``
        (default 0.1).
    eta_f : float, optional
        Evaluation-error budget as a fraction of ``pred``, i.e. the driver
        requests ``|f_err| ≤ eta_f · pred`` (default 0.25).
    cg_tol_start : float, optional
        Initial state-solve relative tolerance (default: the
        homogenization's fixed ``cg_tol``). The accuracy control tightens it
        as needed -- start coarse.
    cg_tol_min : float, optional
        Floor for the state-solve tolerance (default 1e-10), raised to the
        precision floor of the problem's dtype (see
        :func:`muTopOpt.precision.cg_rtol_floor`).
    hv_rtol : float, optional
        Relative tolerance of the two extra solves in each Hessian-vector
        product. ``None`` (default) adapts it to the Steihaug CG that asks
        for the product (see :func:`hessian_rtol`): loose while the
        trust-region subproblems are solved coarsely, tighter as the outer
        iteration converges. A number fixes it. Either way it may be loose:
        the Hessian only shapes the trust-region model, and the method's
        convergence theory requires only a bounded model Hessian.
    callback : callable, optional
        ``callback(it, rho, problem.last)`` after each accepted iterate;
        ``rho`` is in the fields' array module (a cupy array on a GPU).
    disp : bool, optional
        Let NuMPI print its per-iteration table (default False).

    Returns
    -------
    rho_opt : np.ndarray
        Final density on the host, same shape as ``rho0``.
    info : dict
        ``success``, ``message``, ``nit`` (iterations including rejected
        steps), ``objective``, ``max_grad`` (in ``ĝ`` units), ``history``
        (objective of every accepted iterate), ``nb_hessp``,
        ``delta_history``/``rho_history`` (RMS radius and ratio ``r`` per
        iteration) and ``final_cg_rtol``.

    Raises
    ------
    ImportError
        If the installed NuMPI lacks ``tr_newton_bounded``.
    ValueError
        If ``problem`` was not built with Hessian support.
    """
    try:
        from NuMPI.Optimization import tr_newton_bounded
    except ImportError as err:
        raise ImportError(
            "optimize_trust_region needs NuMPI with tr_newton_bounded "
            "(NuMPI > 1.5.1 / current NuMPI main)") from err

    if not getattr(problem, "hessian", False):
        raise ValueError(
            "optimize_trust_region needs a problem with Hessian support; "
            "construct StressTargetProblem(..., hessian=True)")

    if cg_tol_start is None:
        cg_tol_start = getattr(problem.h, "cg_tol", 1e-8)
    # The default floor (1e-10) is four orders below what float32 can reach;
    # the clamp keeps the accuracy control from requesting tolerances at
    # which only the recursive CG residual, not the true one, decreases.
    cg_tol_min = max(cg_tol_min, _problem_rtol_floor(problem))
    cg_tol_start = max(cg_tol_start, cg_tol_min)
    # Reuse AdaptiveInnerTolerance purely as the rtol holder the problem
    # already knows how to read (`.current`) -- the trust-region accuracy
    # control below, not the forcing-term/ratchet logic, drives it (advance()
    # is never called).
    controller = AdaptiveInnerTolerance(cg_tol_start, cg_tol_min,
                                        bounds=bounds)
    problem.inner_tolerance = controller

    history = []
    # Hessian-vector-product CG work accumulated since the last accepted
    # iterate. The Steihaug solves are the bulk of the trust-region cost but
    # happen outside objective_and_gradient, so they are invisible to
    # ``problem.last['cg_iters']``; tally them here and attribute them to the
    # outer iterate whose step they computed.
    hv_stats = {"cg": 0, "hessp": 0, "rtol": None}

    def fun(x):
        return problem.objective_and_gradient(x)

    def hessp(x, v, tol=None):
        # The Hessian is evaluated around the cached linearization point;
        # re-prime it if the last evaluation was a rejected trial iterate.
        problem.ensure_state(x)
        rtol = hessian_rtol(tol) if hv_rtol is None else hv_rtol
        hv = problem.hessian_vector_product(v, rtol=rtol)
        hv_stats["cg"] += int(sum(problem.last_hv_cg_iters))
        hv_stats["hessp"] += 1
        hv_stats["rtol"] = rtol
        return hv

    def fun_error():
        # Computable first-order bound on the evaluation error of the
        # (Lagrangian-corrected) objective: the adjoint-weighted residuals.
        return float(sum(abs(c) for c in problem.last.get("corrections", [])))

    def request_accuracy(target):
        # Consistent objective => f_err ~ C * rtol^2: shrink rtol by the
        # square root of the requested reduction (x0.5 safety), monotone,
        # floored.
        err = fun_error()
        rtol = controller.current
        if err > 0.0 and target > 0.0:
            rtol = rtol * 0.5 * np.sqrt(target / err)
        else:
            rtol = rtol * 0.1
        controller.current = float(max(cg_tol_min,
                                       min(controller.current, rtol)))

    def _cb(x):
        # Attribute the Hessian-vector-product work since the last accepted
        # iterate to this one, then reset the tally.
        problem.last["hv_cg_iters"] = hv_stats["cg"]
        problem.last["nb_hessp"] = hv_stats["hessp"]
        problem.last["hv_rtol"] = hv_stats["rtol"]
        hv_stats["cg"] = 0
        hv_stats["hessp"] = 0
        history.append(problem.last.get("objective"))
        if callback is not None:
            return callback(len(history), x, problem.last)
        return False

    # NuMPI compares the raw per-variable gradient; gtol/max_grad are in
    # mesh-invariant ĝ units (see _gradient_scale).
    gscale = _gradient_scale(problem)
    res = tr_newton_bounded(
        # On the fields' device: the trust region runs in the array module
        # of x0 (cupy on a GPU), so the design, gradient and Hessian
        # products never leave the device; only scalars come to the host.
        fun, problem.h.xp.asarray(np.asarray(rho0, dtype=float)), hessp,
        jac=None,
        bounds_lo=bounds[0], bounds_hi=bounds[1],
        gtol=gtol * gscale, maxiter=maxiter,
        delta0=delta0, delta_max=delta_max, eta=eta,
        fun_error=fun_error, request_accuracy=request_accuracy, eta_f=eta_f,
        hessp_tol=hv_rtol is None,
        comm=comm, callback=_cb, disp=disp,
    )
    info = {
        "success": bool(res.success),
        "message": res.message,
        "nit": int(res.nit),
        "objective": float(res.fun),
        "max_grad": float(res.get("max_grad", np.nan)) / gscale,
        "history": history,
        "nb_hessp": int(res.get("nb_hessp", 0)),
        "delta_history": res.get("delta_history"),
        "rho_history": res.get("rho_history"),
        "final_cg_rtol": controller.current,
    }
    return problem.h.to_host(res.x), info


def optimize_lbfgs(problem, rho0, maxiter=200, gtol=2.5, ftol=1e-9,
                   bounds=(0.0, 1.0), callback=None, cg_tol_start=None,
                   cg_tol_min=None, cg_forcing_c=1.0, cg_forcing_exp=1.0,
                   cg_stall_rel=1e-2, cg_stall_shrink=0.3):
    """Minimize ``problem`` from ``rho0`` with SciPy's L-BFGS-B (serial
    only). Returns ``(rho_opt, info)``.

    ``gtol`` (default 2.5) is in the same mesh-invariant units as
    :func:`optimize_bounded_lbfgs` (the volume-fraction derivative
    ``ĝ = (V/V_e) ∂f/∂ρ``) and is passed to SciPy's projected-gradient test;
    ``ftol`` (default 1e-9) is SciPy's relative-reduction test.

    ``cg_tol_start`` (and the other ``cg_*`` arguments) enable the same
    adaptive inner CG tolerance as :func:`optimize_bounded_lbfgs`; see
    :class:`AdaptiveInnerTolerance`.

    ``info`` holds ``success``, ``message``, ``nit``, ``objective``,
    ``history`` and ``cg_rtol_history`` (no ``max_grad``)."""
    from scipy.optimize import minimize

    shape = np.asarray(rho0).shape
    history = []
    inner_tol = _make_inner_tolerance(
        problem, cg_tol_start, cg_tol_min, cg_forcing_c, cg_forcing_exp,
        cg_stall_rel, cg_stall_shrink, bounds)

    def fun(x):
        f, g = problem.objective_and_gradient(x.reshape(shape))
        return float(f), np.asarray(g, dtype=float).ravel()

    def _cb(xk):
        if inner_tol is not None:
            inner_tol.advance()
        history.append(problem.last.get("objective"))
        if callback is not None:
            return callback(len(history), xk.reshape(shape), problem.last)
        return False

    # SciPy's pgtol compares the raw per-variable gradient; gtol is in
    # mesh-invariant ĝ units (see _gradient_scale).
    res = minimize(
        fun, np.asarray(rho0, dtype=float).ravel(),
        jac=True, method="L-BFGS-B",
        bounds=[bounds] * int(np.prod(shape)),
        options={"maxiter": maxiter, "gtol": gtol * _gradient_scale(problem),
                 "ftol": ftol},
        callback=_cb,
    )
    info = {
        "success": res.success,
        "message": res.message,
        "nit": res.nit,
        "objective": float(res.fun),
        "history": history,
        "cg_rtol_history": (inner_tol.history if inner_tol is not None
                            else None),
    }
    return res.x.reshape(shape), info
