"""
Adjoint-corrected (Lagrangian) objective: L = f + Σ_Γ λ_Γᵀ (K u_Γ - b_Γ).

The correction cancels the first-order effect of truncated CG solves on the
objective, so the reported value is second-order accurate in the solve error
and consistent with the adjoint gradient (∂L/∂ρ). Tested:

  1. superconvergence: at loose cg_tol the corrected objective is far closer
     to the tight-tolerance objective than the raw one,
     |L(loose) - f(tight)| < 0.1 |f_raw(loose) - f(tight)|;
  2. the corrected error falls superlinearly with the CG tolerance;
  3. the correction vanishes at tight tolerance;
  4. the corrected (objective, gradient) pair stays finite-difference
     consistent at a loose tolerance (cg_tol = 1e-4);
  5. an optimization at loose tolerance reaches the same design quality as
     one at tight tolerance.
"""

import numpy as np
import pytest

from muTopOpt import (
    Homogenization,
    NodalPhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases
from muTopOpt.optimize import initial_density, optimize_bounded_lbfgs

from mpi_helpers import (
    local,
    local_index,
    optimizer_comm,
    perturbed,
)


def _problem(dim, n, comm, cg_tol, consistent, design="nodal", seed=0):
    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0, void_ratio=1e-3)
    homog = Homogenization((n,) * dim, material, comm=comm, element="p1",
                           cg_tol=cg_tol, cg_maxiter=5000)
    cases = target_load_cases(
        dim, isotropic_stiffness_tensor(dim, K=0.08, G=0.03), magnitude=0.01)
    reg = NodalPhaseFieldRegularization(homog)
    return StressTargetProblem(homog, cases, regularization=reg,
                               design=design, consistent_objective=consistent)


def _smooth_rho(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    spec = np.fft.fftn(rng.standard_normal((n,) * dim))
    for ax in range(dim):
        f = np.fft.fftfreq(n)
        g = np.exp(-2.0 * np.pi**2 * (0.12 * n) ** 2 * f**2)
        spec *= g.reshape([-1 if d == ax else 1 for d in range(dim)])
    rho = np.fft.ifftn(spec).real
    rho -= rho.mean()
    rho /= max(rho.std(), 1e-30)
    return np.clip(0.5 + 0.3 * rho, 0.05, 0.95)


def _objective(problem, rho):
    """Objective at the global design ``rho`` (each rank passes its slab)."""
    return problem.objective_and_gradient(local(problem.h, rho))[0]


def test_superconvergence_of_corrected_objective(comm):
    """|L(loose) - f(tight)| must be much smaller than |f_raw(loose) -
    f(tight)|: the correction removes the first-order solve-error effect.
    (Measured on this problem: ratio 0.03 at cg_tol=1e-2, 0.009 at 1e-3.)"""
    n = 16
    rho = _smooth_rho(n, 2)
    f_exact = _objective(_problem(2, n, comm, 1e-12, True), rho)

    loose = 1e-2
    f_corr = _objective(_problem(2, n, comm, loose, True), rho)
    f_raw = _objective(_problem(2, n, comm, loose, False), rho)

    err_corr = abs(f_corr - f_exact)
    err_raw = abs(f_raw - f_exact)
    assert err_raw > 0  # the loose solve must actually be inexact
    # Second-order vs first-order: demand at least a factor 10.
    assert err_corr < 0.1 * err_raw, (
        f"corrected error {err_corr:.3e} not << raw error {err_raw:.3e}")


def test_corrected_error_scales_superlinearly(comm):
    """The corrected objective error must fall superlinearly with the CG
    tolerance (second-order accuracy): a 10x tighter tolerance must reduce it
    by much more than 10x."""
    n = 16
    rho = _smooth_rho(n, 2)
    f_exact = _objective(_problem(2, n, comm, 1e-12, True), rho)

    def err(tol):
        return abs(_objective(_problem(2, n, comm, tol, True), rho) - f_exact)

    e_loose, e_tight = err(1e-2), err(1e-3)
    assert e_tight < 0.02 * e_loose, (
        f"error {e_loose:.3e} -> {e_tight:.3e} over a 10x tolerance "
        "reduction is not superlinear")


def test_correction_vanishes_at_tight_tolerance(comm):
    """At cg_tol = 1e-12 every per-load-case correction λ_Γᵀ (K u_Γ - b_Γ)
    (three load cases in 2D) is below 1e-9 of the objective."""
    n = 12
    rho = _smooth_rho(n, 2, seed=1)
    problem = _problem(2, n, comm, 1e-12, True)
    _objective(problem, rho)
    corrections = problem.last["corrections"]
    assert len(corrections) == 3
    scale = abs(problem.last["objective"]) + 1e-30
    assert max(abs(c) for c in corrections) < 1e-9 * scale


def _fd_mismatch(problem, rho, sample, d=1e-5):
    """Worst relative mismatch between the assembled gradient and central
    finite differences of the reported objective,
    max_i |g_i - fd_i| / max(|g_i|, |fd_i|) with
    fd_i = (f(rho + d e_i) - f(rho - d e_i)) / (2 d), over the global pixels
    ``sample`` of the global design ``rho``."""
    h = problem.h
    x = local(h, rho)
    _, g = problem.objective_and_gradient(x)
    worst = 0.0
    for gi in sample:
        fp, _ = problem.objective_and_gradient(perturbed(h, x, gi, d))
        fm, _ = problem.objective_and_gradient(perturbed(h, x, gi, -d))
        fd = (fp - fm) / (2 * d)
        idx = local_index(h, gi)
        if idx is not None:
            denom = max(abs(g[idx]), abs(fd), 1e-12)
            worst = max(worst, abs(g[idx] - fd) / denom)
    return h.comm.max(worst)
def test_fd_consistency_at_loose_tolerance(comm):
    """The (objective, gradient) pair must remain usably FD-consistent at a
    loose inner tolerance. Note the correction makes the *value* second-order
    accurate and smooth; the gradient itself keeps a first-order O(delta)
    error (the K(lambda - lambda*) du/drho term), so the bound here is
    modest -- the decisive functional test is
    test_loose_tolerance_optimization_quality."""
    n = 12
    rho = _smooth_rho(n, 2, seed=2)
    sample = [(0, 0), (3, 7), (8, 2), (n - 1, n - 1)]

    mismatch = _fd_mismatch(_problem(2, n, comm, 1e-4, True), rho, sample)
    assert mismatch < 2e-2, (
        f"FD mismatch {mismatch:.3e} at cg_tol=1e-4 too large")


def test_loose_tolerance_optimization_quality(comm):
    """A short optimization at cg_tol = 1e-3 (corrected) must reach the same
    objective as at 1e-8, judged by re-evaluating both final designs with
    tight solves."""
    n = 16
    rho0 = initial_density((n,) * 2, kind="uniform", volume_fraction=0.5)
    # Perturb so the runs don't start on the double-well ridge.
    rng = np.random.default_rng(3)
    rho0 = np.clip(rho0 + 0.05 * rng.standard_normal(rho0.shape), 0, 1)

    results = {}
    for label, tol in [("tight", 1e-8), ("loose", 1e-3)]:
        problem = _problem(2, n, comm, tol, True)
        rho_opt, info = optimize_bounded_lbfgs(
            problem, local(problem.h, rho0), comm=optimizer_comm(comm),
            maxiter=40)
        results[label] = rho_opt

    judge = _problem(2, n, comm, 1e-12, True)
    f_tight, _ = judge.objective_and_gradient(results["tight"])
    f_loose, _ = judge.objective_and_gradient(results["loose"])
    # Same-quality optimization: within a few percent of each other.
    assert f_loose < f_tight * 1.05 + 1e-12, (
        f"loose-tolerance run degraded: {f_loose:.6f} vs {f_tight:.6f}")
