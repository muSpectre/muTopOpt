"""Unit tests for the SIMP conductivity interpolation and its derivatives."""

import numpy as np

from muTopOpt import SimpConductivity


def test_endpoints():
    m = SimpConductivity(kappa_solid=2.0, penalty=3.0, void_ratio=1e-3)
    k1 = m.kappa(np.array([1.0]))
    k0 = m.kappa(np.array([0.0]))
    np.testing.assert_allclose(k1, [2.0], rtol=1e-12)
    np.testing.assert_allclose(k0, [2.0 * 1e-3], rtol=1e-12)


def test_derivative_matches_fd():
    m = SimpConductivity(kappa_solid=1.7, penalty=3.0, void_ratio=1e-2)
    rho = np.linspace(0.1, 0.9, 9)
    dkappa = m.dkappa(rho)
    d = 1e-6
    k_p = m.kappa(rho + d)
    k_m = m.kappa(rho - d)
    np.testing.assert_allclose(dkappa, (k_p - k_m) / (2 * d), rtol=1e-5)


def test_second_derivative_matches_fd():
    m = SimpConductivity(kappa_solid=1.3, penalty=2.5, void_ratio=1e-2)
    rho = np.linspace(0.1, 0.9, 9)
    d2kappa = m.d2kappa(rho)
    d = 1e-4
    dk_p = m.dkappa(rho + d)
    dk_m = m.dkappa(rho - d)
    np.testing.assert_allclose(d2kappa, (dk_p - dk_m) / (2 * d), rtol=1e-3)
