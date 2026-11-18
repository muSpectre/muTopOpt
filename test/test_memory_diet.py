"""
Device-memory savings that must not change any result:

1. the scratch fields of the residual-stopped CG (``K u``, best iterate)
   exist only once a residual-stopped solve needed them; a run with
   ``cg_stop="energy"`` never allocates them;
2. ``SimpMaterial.lame``/``dlame`` with ``out=`` give bit for bit the values
   of the plain call;
3. the material written straight into double-precision fields by
   ``set_density`` equals the plain interpolation, and so does the stiffness
   scale computed without an ``|·|`` temporary.
"""
import numpy as np
import pytest

from muTopOpt.homogenization import Homogenization, conjugate_gradients_energy
from muTopOpt.material import SimpMaterial

SCRATCH = ("to_Ku", "to_cg_best")


def _homogenization(**options):
    material = SimpMaterial(1.0, 0.3, penalty=3, void_ratio=1e-3)
    return Homogenization((8, 8, 8), material, **options)


def _solve(h):
    rho = np.random.default_rng(0).uniform(0.0, 1.0, h.nb_pixels)
    h.set_density(rho)
    u = h.vector_field("u")
    E = np.diag([1.0, 0.0, 0.0])
    h.solve_macro(E, u)
    h.solve_macro(E, u, warm_start=True)  # the warm-start guard uses K u


@pytest.mark.skipif(conjugate_gradients_energy is None,
                    reason="needs muGrid's energy-stopped CG")
def test_energy_cg_allocates_no_scratch(comm):
    h = _homogenization(comm=comm, cg_stop="energy")
    _solve(h)
    assert not any(name in h._scratch for name in SCRATCH)


def test_residual_cg_allocates_its_scratch_once(comm):
    h = _homogenization(comm=comm, cg_stop="residual")
    _solve(h)
    assert all(name in h._scratch for name in SCRATCH)
    # the same field on every access
    assert h._Ku is h._Ku and h._x_best is h._x_best


@pytest.mark.parametrize("method", ["lame", "dlame"])
def test_material_out_is_bit_identical(method):
    material = SimpMaterial(1.3, 0.28, penalty=2.0, void_ratio=1e-3)
    rho = np.random.default_rng(1).uniform(0.0, 1.0, (5, 6, 7))
    plain = getattr(material, method)(rho)
    out = (np.empty_like(rho), np.empty_like(rho))
    returned = getattr(material, method)(rho, out=out)
    for expected, value, target in zip(plain, returned, out):
        assert value is target
        np.testing.assert_array_equal(value, expected)


def test_set_density_material_and_scale(comm):
    h = _homogenization(comm=comm)
    rho = np.random.default_rng(2).uniform(0.0, 1.0, h.nb_pixels)
    h.set_density(rho)
    lam, mu = h.material.lame(rho)
    np.testing.assert_array_equal(h.to_host(h.lam.p), lam)
    np.testing.assert_array_equal(h.to_host(h.mu.p), mu)
    scale = h.comm.max(max(float(np.abs(lam).max()),
                           float(2.0 * np.abs(mu).max())))
    assert h.mat_scale == scale
