"""Unit tests for the SIMP material interpolation and its derivatives."""

import numpy as np

from muTopOpt import SimpMaterial, lame_from_E_nu


def test_endpoints():
    m = SimpMaterial(E_solid=2.0, nu=0.3, penalty=3.0, void_ratio=1e-3)
    lam1, mu1 = m.lame(np.array([1.0]))
    lam0, mu0 = m.lame(np.array([0.0]))
    ls, ms = lame_from_E_nu(2.0, 0.3)
    np.testing.assert_allclose([lam1[0], mu1[0]], [ls, ms], rtol=1e-12)
    np.testing.assert_allclose([lam0[0], mu0[0]], [ls * 1e-3, ms * 1e-3], rtol=1e-12)


def test_derivative_matches_fd():
    m = SimpMaterial(E_solid=1.7, nu=0.25, penalty=3.0, void_ratio=1e-2)
    rho = np.linspace(0.1, 0.9, 9)
    dlam, dmu = m.dlame(rho)
    d = 1e-6
    lam_p, mu_p = m.lame(rho + d)
    lam_m, mu_m = m.lame(rho - d)
    np.testing.assert_allclose(dlam, (lam_p - lam_m) / (2 * d), rtol=1e-5)
    np.testing.assert_allclose(dmu, (mu_p - mu_m) / (2 * d), rtol=1e-5)
