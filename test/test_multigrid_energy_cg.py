"""
Tests for the multigrid preconditioner and the energy-error CG stopping test
of :class:`muTopOpt.homogenization.Homogenization`.
"""
import numpy as np
import pytest

import muGrid

from muTopOpt.homogenization import Homogenization
from muTopOpt.material import SimpMaterial

pytest.importorskip("muGrid.Multigrid")


def density(nb_grid_pts, sharpness=4.0):
    """A smooth two-phase design: solid where the sum of cosines is positive,
    with an interface a few elements wide."""
    axes = [(np.arange(n) + 0.5) / n for n in nb_grid_pts]
    grids = np.meshgrid(*axes, indexing="ij")
    level = sum(np.cos(2 * np.pi * g) for g in grids)
    return 0.5 + 0.5 * np.tanh(sharpness * level)


def unit_strain(dim, i, j):
    E = np.zeros((dim, dim))
    E[i, j] = E[j, i] = 1.0 if i == j else 0.5
    return E


def solve(nb_grid_pts, material=None, strain=(0, 0), **options):
    """Homogenised stress, CG iterations and the solver for one load case."""
    dim = len(nb_grid_pts)
    material = material or SimpMaterial(1.0, 0.3, penalty=3, void_ratio=1e-3)
    h = Homogenization(nb_grid_pts, material, **options)
    h.set_density(density(nb_grid_pts))
    u = h.vector_field("u")
    E = unit_strain(dim, *strain)
    h.solve_macro(E, u)
    return h.homogenized_stress(u, E), h.last_cg_iters, h, u, E


COARSENINGS = ["galerkin", "rediscretised"]


@pytest.mark.parametrize("coarsening", COARSENINGS)
@pytest.mark.parametrize("nb_grid_pts", [(32, 32), (16, 16, 16)])
@pytest.mark.parametrize("precision", ["double", "single"])
def test_multigrid_gives_the_stress_of_green_jacobi(nb_grid_pts, precision,
                                                    coarsening):
    """Same homogenised stress as a tight Green-Jacobi solve, in fewer
    iterations, for a contrast of 1000, with Galerkin and with re-discretised
    coarse levels."""
    reference, iterations_reference, *_ = solve(
        nb_grid_pts, preconditioner="green-jacobi", cg_tol=1e-10)
    stress, iterations, *_ = solve(
        nb_grid_pts, preconditioner="multigrid", cg_tol=1e-6,
        cg_stop="energy", multigrid_precision=precision,
        multigrid_coarsening=coarsening)
    # an energy error of 1e-6 is an error of 1e-12 in the energy; the stress
    # is linear in the solution, so its error is of first order
    np.testing.assert_allclose(stress, reference,
                               atol=1e-5 * np.abs(reference).max())
    assert iterations < iterations_reference / 2


def test_iterations_do_not_grow_with_the_grid():
    counts = [solve((n, n), preconditioner="multigrid", cg_tol=1e-6,
                    cg_stop="energy")[1] for n in (16, 32, 64)]
    assert max(counts) - min(counts) <= 3
    assert max(counts) < 25


@pytest.mark.parametrize("stop", ["residual", "energy"])
def test_multigrid_with_either_stopping_test(stop):
    reference, *_ = solve((32, 32), strain=(0, 1),
                          preconditioner="green-jacobi", cg_tol=1e-10)
    stress, *_ = solve((32, 32), strain=(0, 1), preconditioner="multigrid",
                       cg_tol=1e-7, cg_stop=stop)
    np.testing.assert_allclose(stress, reference,
                               atol=1e-5 * np.abs(reference).max())


@pytest.mark.parametrize("coarsening", COARSENINGS)
def test_material_with_independent_lame_parameters(coarsening):
    """A void phase that is not a multiple of the solid: the multigrid then
    uses one family of element matrices for λ and one for μ (on the coarse
    levels too, when they are re-discretised)."""
    material = SimpMaterial(1.0, 0.3, penalty=3, void_ratio=1e-3)
    material.lam_void *= 5.0        # another Poisson ratio in the void
    reference, *_ = solve((32, 32), material,
                          preconditioner="green-jacobi", cg_tol=1e-10)
    stress, _, h, *_ = solve((32, 32), material, preconditioner="multigrid",
                             cg_tol=1e-7, cg_stop="energy",
                             multigrid_coarsening=coarsening)
    assert h._mg_scales is not None
    np.testing.assert_allclose(stress, reference,
                               atol=1e-5 * np.abs(reference).max())


def test_energy_stop_with_green_jacobi():
    """The stopping test is independent of the preconditioner."""
    reference, *_ = solve((32, 32), preconditioner="green-jacobi",
                          cg_tol=1e-10)
    stress, *_ = solve((32, 32), preconditioner="green-jacobi", cg_tol=1e-6,
                       cg_stop="energy")
    np.testing.assert_allclose(stress, reference,
                               atol=1e-5 * np.abs(reference).max())


def test_energy_stop_returns_the_residual():
    """The residual field of an energy-stopped solve is b − K x of the
    returned iterate, as the adjoint-corrected objective needs it."""
    nb_grid_pts = (32, 32)
    h = Homogenization(nb_grid_pts, SimpMaterial(1.0, 0.3, void_ratio=1e-3),
                       preconditioner="multigrid", cg_tol=1e-3,
                       cg_stop="energy")
    h.set_density(density(nb_grid_pts))
    u, residual = h.vector_field("u"), h.vector_field("residual")
    E = unit_strain(2, 0, 0)
    h.solve_macro(E, u, residual=residual)
    Ku = h.vector_field("Ku")
    h._hessp(u, Ku)
    true_residual = h._rhs.p - Ku.p
    assert np.abs(true_residual).max() > 0          # a loose solve
    np.testing.assert_allclose(residual.p, true_residual,
                               atol=1e-9 * np.abs(h._rhs.p).max())


def test_warm_start_needs_fewer_iterations():
    """Solving again from the solution of a slightly different design."""
    nb_grid_pts = (32, 32)
    h = Homogenization(nb_grid_pts, SimpMaterial(1.0, 0.3, void_ratio=1e-3),
                       preconditioner="multigrid", cg_tol=1e-6,
                       cg_stop="energy")
    rho = density(nb_grid_pts)
    E = unit_strain(2, 0, 0)
    u = h.vector_field("u")
    h.set_density(rho)
    h.solve_macro(E, u)
    cold = h.last_cg_iters
    h.set_density(np.clip(rho + 1e-3 * np.cos(40 * rho), 0, 1))
    h.solve_macro(E, u, warm_start=True)
    warm = h.last_cg_iters
    h.solve_macro(E, u)
    assert warm < cold
    assert h.last_cg_iters >= cold - 1              # cold again: as before


@pytest.mark.parametrize("coarsening", COARSENINGS)
def test_multigrid_setup_follows_the_design(coarsening):
    """A second design gets its own coarse operators: solving it with the
    refreshed multigrid takes as few iterations as a fresh solver does."""
    nb_grid_pts = (32, 32)
    options = dict(preconditioner="multigrid", cg_tol=1e-6, cg_stop="energy",
                   multigrid_coarsening=coarsening)
    material = SimpMaterial(1.0, 0.3, void_ratio=1e-3)
    E = unit_strain(2, 0, 0)
    second = density(nb_grid_pts, sharpness=1.0)[::-1].copy()

    h = Homogenization(nb_grid_pts, material, **options)
    u = h.vector_field("u")
    h.set_density(density(nb_grid_pts))
    h.solve_macro(E, u)
    h.set_density(second)
    h.solve_macro(E, u)
    refreshed = h.last_cg_iters

    fresh = Homogenization(nb_grid_pts, material, **options)
    fresh.set_density(second)
    v = fresh.vector_field("u")
    fresh.solve_macro(E, v)
    assert refreshed == fresh.last_cg_iters
    np.testing.assert_allclose(u.p, v.p, atol=1e-9 * np.abs(v.p).max())


def test_unsupported_combinations_raise():
    material = SimpMaterial(1.0, 0.3)
    with pytest.raises(ValueError, match="cg_stop"):
        Homogenization((16, 16), material, cg_stop="error")
    h = Homogenization((16, 16), material, preconditioner="multigrid",
                       dtype=np.float32)
    with pytest.raises(ValueError, match="float64"):
        h.set_density(density((16, 16)))
    h = Homogenization((16, 16), SimpMaterial(1.0, 0.3, void_ratio=0.0),
                       preconditioner="multigrid")
    with pytest.raises(ValueError, match="void"):
        h.set_density(density((16, 16)))
    with pytest.raises(ValueError, match="multigrid_coarsening"):
        Homogenization((16, 16), material, preconditioner="multigrid",
                       multigrid_coarsening="algebraic")
    # re-discretised levels exist for Q1 only
    h = Homogenization((16, 16), material, preconditioner="multigrid",
                       element="p1", multigrid_coarsening="rediscretised")
    with pytest.raises(ValueError, match="q1"):
        h.set_density(density((16, 16)))


@pytest.mark.gpu
def test_gpu_matches_cpu():
    pytest.importorskip("cupy")
    if not muGrid.has_gpu:
        pytest.skip("muGrid without GPU support")
    options = dict(preconditioner="multigrid", cg_tol=1e-6, cg_stop="energy",
                   multigrid_precision="single")
    cpu, iterations_cpu, *_ = solve((16, 16, 16), **options)
    gpu, iterations_gpu, *_ = solve((16, 16, 16), device="gpu", **options)
    assert abs(iterations_gpu - iterations_cpu) <= 1
    np.testing.assert_allclose(gpu, cpu, atol=1e-6 * np.abs(cpu).max())
