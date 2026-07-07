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
                           callback=None):
    """Minimize ``problem`` from ``rho0`` with NuMPI's MPI-distributed,
    box-constrained L-BFGS. Returns ``(rho_opt, info)``.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm, optional
        The communicator over which the density is distributed (the *same*
        decomposition as the muGrid fields, typically ``MPI.COMM_WORLD``).
        ``None`` runs serially. Density stays in ``bounds`` by projection.
    """
    from NuMPI.Optimization import l_bfgs_bounded

    history = []

    # The local density grid is passed as-is: NuMPI's l_bfgs_bounded handles an
    # n-D x0 (keeping its iterate/gradient/history flat internally) and returns
    # the result in the same shape.
    def fun(x):
        return problem.objective_and_gradient(x)

    def _cb(x):
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
    }
    return np.asarray(res.x), info


def optimize_lbfgs(problem, rho0, maxiter=200, gtol=1e-5, ftol=1e-9,
                   bounds=(0.0, 1.0), callback=None):
    """Minimize ``problem`` from ``rho0`` with L-BFGS-B. Returns
    ``(rho_opt, info)``."""
    from scipy.optimize import minimize

    shape = np.asarray(rho0).shape
    history = []

    def fun(x):
        f, g = problem.objective_and_gradient(x.reshape(shape))
        return float(f), np.asarray(g, dtype=float).ravel()

    def _cb(xk):
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
    }
    return res.x.reshape(shape), info
