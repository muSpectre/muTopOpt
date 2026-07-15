"""Unit tests for the target-feasibility bounds (Voigt / Hashin-Shtrikman)."""

import numpy as np
import pytest

from muTopOpt.bounds import (
    hashin_shtrikman_upper,
    solid_moduli,
    target_feasibility,
    voigt_upper,
)


@pytest.mark.parametrize("dim", [2, 3])
def test_bound_limits_and_ordering(dim):
    K0, G0 = solid_moduli(dim, 1.0, 0.3)
    # phi = 1 recovers the solid, phi = 0 the void
    np.testing.assert_allclose(
        hashin_shtrikman_upper(dim, 1.0, K0, G0), [K0, G0], rtol=1e-12)
    np.testing.assert_allclose(
        hashin_shtrikman_upper(dim, 0.0, K0, G0), [0.0, 0.0], atol=1e-12)
    # Hashin-Shtrikman is monotone in phi and never exceeds Voigt
    prev = (0.0, 0.0)
    for phi in np.linspace(0.05, 0.95, 19):
        hs = hashin_shtrikman_upper(dim, phi, K0, G0)
        vt = voigt_upper(dim, phi, K0, G0)
        assert hs[0] <= vt[0] and hs[1] <= vt[1]
        assert hs[0] > prev[0] and hs[1] > prev[1]
        prev = hs


def test_feasibility_inverts_bound():
    dim = 3
    K0, G0 = solid_moduli(dim, 1.0, 0.3)
    feas = target_feasibility(dim, 0.3, 0.15, 1.0, 0.3)
    K_at, _ = hashin_shtrikman_upper(dim, feas["phi_hs_K"], K0, G0)
    _, G_at = hashin_shtrikman_upper(dim, feas["phi_hs_G"], K0, G0)
    np.testing.assert_allclose(K_at, 0.3, rtol=1e-6)
    np.testing.assert_allclose(G_at, 0.15, rtol=1e-6)


def test_comfortable_target_no_warnings():
    # K, G reachable around 55-60% solid: inside the [0.2, 0.8] comfort range
    feas = target_feasibility(3, 0.3, 0.15, 1.0, 0.3)
    assert 0.2 < feas["phi_min"] < 0.8
    assert feas["warnings"] == []


def test_auxetic_high_g_target_warns_high_density():
    # E=0.5, nu=-0.3 asks for G = 0.93 G0: needs > 90% solid per the bound
    target_K, target_G = solid_moduli(3, 0.5, -0.3)
    feas = target_feasibility(3, target_K, target_G, 1.0, 0.3)
    assert feas["phi_hs_G"] > 0.9
    assert feas["phi_min"] == feas["phi_hs_G"]
    assert len(feas["warnings"]) == 1
    assert "at least" in feas["warnings"][0]


def test_target_exceeding_solid_is_unreachable():
    feas = target_feasibility(3, 1.0, 1.0, 1.0, 0.3)  # K0 = 0.83, G0 = 0.38
    assert np.isinf(feas["phi_min"])
    assert len(feas["warnings"]) == 2
    assert all("unreachable" in w for w in feas["warnings"])


def test_tiny_target_warns_percolation():
    feas = target_feasibility(3, 1e-3, 1e-3, 1.0, 0.3)
    assert feas["phi_min"] < 0.2
    assert len(feas["warnings"]) == 1
    assert "percolate" in feas["warnings"][0]


@pytest.mark.parametrize("dim", [2, 3])
def test_solid_moduli_convention(dim):
    # K = lambda + 2 mu / dim: 3D bulk modulus, 2D plane-strain area modulus
    from muTopOpt import lame_from_E_nu

    lam, mu = lame_from_E_nu(1.0, 0.3)
    K, G = solid_moduli(dim, 1.0, 0.3)
    np.testing.assert_allclose(K, lam + 2.0 * mu / dim, rtol=1e-12)
    np.testing.assert_allclose(G, mu, rtol=1e-12)
