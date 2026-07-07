#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Gradient-based optimization drivers for the element-wise density problem.

:func:`optimize_bounded_lbfgs` (the default) wraps NuMPI's **MPI-distributed**
bound-constrained L-BFGS (``l_bfgs_bounded``): the density is kept in ``[0, 1]``
by box projection, and every reduction goes through NuMPI so the optimizer runs
correctly across ranks on the same domain decomposition as the muGrid fields.
Its MPI contract -- ``x``/gradient/bounds are local per-rank slices, the scalar
energy is globally reduced -- is exactly what :meth:`StressTargetProblem.
objective_and_gradient` already provides.

:func:`optimize_lbfgs` wraps SciPy's serial L-BFGS-B and is kept as a
dependency-light alternative for single-process runs.

Both take a problem object providing ``objective_and_gradient(rho) -> (f, grad)``
with ``rho`` shaped like :attr:`Homogenization.nb_pixels`.
"""

import numpy as np


class AdaptiveInnerTolerance:
    """Eisenstat--Walker-style *forcing term* coupling the inner CG tolerance
    to the outer optimizer's stationarity.

    Each accepted outer iterate sets the inner (forward/adjoint) CG relative
    tolerance to

        rtol = clip( c * ||g_free||^alpha , rtol_min , rtol_start )

    where ``||g_free||`` is the infinity norm of the *box-KKT-masked* gradient
    -- the same stationarity measure ``l_bfgs_bounded`` uses for convergence
    (see NuMPI ``_kkt_residual``). Far from the optimum the gradient is large
    and the inner solves stay coarse (no oversolving); as the outer iterate
    approaches a stationary point the tolerance tightens towards ``rtol_min``.

    **Frozen between iterates.** :meth:`advance` recomputes the tolerance once
    per accepted outer iterate (called from the optimizer callback); within an
    iterate's line search :attr:`current` is constant, so every trial point is
    solved to the *same* accuracy. This keeps the L-BFGS secant pair
    ``y = g_new - g_old`` consistent (both gradients carry the same solve
    accuracy), which is the condition under which quasi-Newton methods tolerate
    inexact gradients (Xie, Byrd & Nocedal 2020; Berahas, Byrd & Nocedal 2019).

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
    L-BFGS cannot certify convergence.
    """

    def __init__(self, rtol_start, rtol_min, c=1.0, alpha=1.0,
                 bounds=(0.0, 1.0)):
        self.rtol_start = float(rtol_start)
        self.rtol_min = float(rtol_min)
        self.c = float(c)
        self.alpha = float(alpha)
        self.bounds = bounds
        self.current = float(rtol_start)
        self._latest_gnorm = None
        # (gnorm, rtol) at each advance, for diagnostics / output.
        self.history = []

    def observe(self, gnorm):
        """Record the box-KKT gradient norm of the most recent evaluation.

        Called on *every* objective/gradient evaluation (including line-search
        trials); it only stores the value -- :attr:`current` is untouched until
        :meth:`advance`."""
        self._latest_gnorm = float(gnorm)

    def advance(self):
        """Recompute :attr:`current` from the last observed gradient norm.

        Invoked once per accepted outer iterate (from the optimizer callback).
        A no-op until the first :meth:`observe`."""
        if self._latest_gnorm is None:
            return self.current
        rtol = self.c * self._latest_gnorm ** self.alpha
        self.current = float(min(self.rtol_start, max(self.rtol_min, rtol)))
        self.history.append((self._latest_gnorm, self.current))
        return self.current


def _make_inner_tolerance(problem, cg_tol_start, cg_tol_min, cg_forcing_c,
                          cg_forcing_exp, bounds):
    """Build and attach an :class:`AdaptiveInnerTolerance` to ``problem`` when
    adaptive coupling is requested (``cg_tol_start`` given), else detach any
    previous controller and return ``None`` (fixed ``Homogenization.cg_tol``).

    ``cg_tol_min`` defaults to the homogenization's fixed ``cg_tol`` -- i.e.
    "start coarse at ``cg_tol_start`` and tighten down to the tolerance you
    would otherwise have used throughout"."""
    if cg_tol_start is None:
        problem.inner_tolerance = None
        return None
    if cg_tol_min is None:
        cg_tol_min = getattr(problem.h, "cg_tol", 1e-8)
    controller = AdaptiveInnerTolerance(
        cg_tol_start, cg_tol_min, c=cg_forcing_c, alpha=cg_forcing_exp,
        bounds=bounds,
    )
    problem.inner_tolerance = controller
    return controller


def initial_density(shape, kind="uniform", volume_fraction=0.5, seed=0,
                    smoothing=2, length=None, grid_spacing=None, contrast=0.5):
    """Build an initial element-wise density.

    ``kind='uniform'`` fills with ``volume_fraction``.

    ``kind='random'`` draws a box-smoothed random field (``smoothing`` sweeps),
    a cheap low-pass start with no controlled length scale.

    ``kind='filtered_random'`` draws white noise and applies a **periodic
    Gaussian filter** of correlation length ``length`` (in the *physical* units
    of the unit cell -- the cell has length 1 by default, converted to pixels
    via ``grid_spacing``, which defaults to ``1/n`` per axis). The field is then
    standardized and mapped to ``clip(volume_fraction + contrast * z, 0, 1)`` so
    it has the requested mean volume fraction with smooth blobs of the chosen
    size -- and no sharp interfaces, which the phase-field regularization then
    sharpens on its own during the optimization (no filter is applied there).

    Choosing ``length`` -- the initial blob size must be **larger** than the
    phase-field interface width the regularization will impose. In the
    Modica-Mortola normalization that width is simply ``eta`` itself (which
    defaults to one grid spacing, ``eta = h``), so pick ``length ~= 2..4 *
    eta`` -- the regularization then *sharpens the blob boundaries* rather
    than dissolving the blobs.
    """
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
            g = np.exp(-2.0 * np.pi**2 * sigma_px**2 * f**2)
            spec *= g.reshape([-1 if d == ax else 1 for d in range(ndim)])
        rho = np.fft.ifftn(spec).real

        rho -= rho.mean()
        std = rho.std()
        if std > 0:
            rho /= std
        return np.clip(float(volume_fraction) + float(contrast) * rho, 0.0, 1.0)
    raise ValueError(f"unknown initial-density kind '{kind}'")


def optimize_bounded_lbfgs(problem, rho0, comm=None, maxiter=200, gtol=1e-5,
                           ftol=0.0, xtol=0.0, bounds=(0.0, 1.0), maxcor=10,
                           callback=None, cg_tol_start=None, cg_tol_min=None,
                           cg_forcing_c=1.0, cg_forcing_exp=1.0):
    """Minimize ``problem`` from ``rho0`` with NuMPI's MPI-distributed,
    box-constrained L-BFGS. Returns ``(rho_opt, info)``.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        The communicator over which the density is distributed (the *same*
        decomposition as the muGrid fields, typically ``MPI.COMM_WORLD``).
        ``None`` runs serially. Density stays in ``bounds`` by projection.
    cg_tol_start : float, optional
        Enable *adaptive* inner CG tolerance: the forward/adjoint solves start
        at this (coarse) relative tolerance and tighten as the outer projected
        gradient shrinks (an Eisenstat--Walker forcing term; see
        :class:`AdaptiveInnerTolerance`). ``None`` (default) keeps the fixed
        ``Homogenization.cg_tol`` used today.
    cg_tol_min : float, optional
        Floor for the adaptive tolerance; defaults to the homogenization's
        ``cg_tol``. Must be small enough that the final gradient error is below
        ``gtol`` (the norm condition), or convergence cannot be certified.
    cg_forcing_c, cg_forcing_exp : float, optional
        Coefficient ``c`` and exponent ``alpha`` in
        ``rtol = c * ||g_free||**alpha`` (defaults 1.0, 1.0). ``alpha=1`` gives
        ``rtol = O(||g||)`` and fast local convergence.
    """
    from NuMPI.Optimization import l_bfgs_bounded

    history = []
    inner_tol = _make_inner_tolerance(
        problem, cg_tol_start, cg_tol_min, cg_forcing_c, cg_forcing_exp, bounds)

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
            callback(len(history), x, problem.last)

    res = l_bfgs_bounded(
        fun, np.asarray(rho0, dtype=float), jac=None,
        bounds_lo=bounds[0], bounds_hi=bounds[1],
        gtol=gtol, ftol=ftol, xtol=xtol, maxiter=maxiter, maxcor=maxcor,
        comm=comm, callback=_cb,
    )
    info = {
        "success": bool(res.success),
        "message": res.message,
        "nit": int(res.nit),
        "objective": float(res.fun),
        "max_grad": float(res.get("max_grad", np.nan)),
        "history": history,
        "cg_rtol_history": (inner_tol.history if inner_tol is not None
                            else None),
    }
    return np.asarray(res.x), info


def optimize_lbfgs(problem, rho0, maxiter=200, gtol=1e-5, ftol=1e-9,
                   bounds=(0.0, 1.0), callback=None, cg_tol_start=None,
                   cg_tol_min=None, cg_forcing_c=1.0, cg_forcing_exp=1.0):
    """Minimize ``problem`` from ``rho0`` with L-BFGS-B. Returns
    ``(rho_opt, info)``.

    ``cg_tol_start`` (and the other ``cg_*`` arguments) enable the same
    adaptive inner CG tolerance as :func:`optimize_bounded_lbfgs`; see
    :class:`AdaptiveInnerTolerance`."""
    from scipy.optimize import minimize

    shape = np.asarray(rho0).shape
    history = []
    inner_tol = _make_inner_tolerance(
        problem, cg_tol_start, cg_tol_min, cg_forcing_c, cg_forcing_exp, bounds)

    def fun(x):
        f, g = problem.objective_and_gradient(x.reshape(shape))
        return float(f), np.asarray(g, dtype=float).ravel()

    def _cb(xk):
        if inner_tol is not None:
            inner_tol.advance()
        history.append(problem.last.get("objective"))
        if callback is not None:
            callback(len(history), xk.reshape(shape), problem.last)

    res = minimize(
        fun, np.asarray(rho0, dtype=float).ravel(),
        jac=True, method="L-BFGS-B",
        bounds=[bounds] * int(np.prod(shape)),
        options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol},
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
