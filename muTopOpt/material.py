#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
SIMP material interpolation for element-wise density topology optimization.

The design variable is an element-wise (per-pixel) density ``rho`` in ``[0, 1]``.
The isotropic stiffness is interpolated with the "solid isotropic material with
penalization" (SIMP) power law between a void and a solid phase,

    C(rho) = C_void + rho**p * (C_solid - C_void),

which -- because both phases are isotropic -- reduces to a per-pixel
interpolation of the Lamé parameters ``lambda`` and ``mu``. This is exactly the
per-pixel material that :class:`muGrid.IsotropicStiffnessOperator` consumes, so
no full stiffness tensor is ever stored.

The void phase is given a small but non-zero stiffness (``void_ratio``) which
keeps the tangent positive definite everywhere; the J-FFT (Green-Jacobi)
preconditioner then converges even in the smooth high-contrast interface, and
``rho`` is free to approach true void.
"""

import numpy as np


def lame_from_E_nu(E, nu):
    """Lamé parameters (lambda, mu) of an isotropic material (plane-strain / 3D
    convention, i.e. the same Hooke's law the stiffness operator assembles)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


class SimpMaterial:
    """Power-law (SIMP) interpolation of isotropic Lamé parameters.

    Parameters
    ----------
    E_solid : float
        Young's modulus of the solid phase.
    nu : float
        Poisson's ratio (shared by both phases).
    penalty : float, optional
        SIMP exponent ``p`` (default 2). ``p > 1`` penalizes intermediate
        densities.
    void_ratio : float, optional
        Stiffness ratio of the void phase, ``E_void / E_solid`` (default 1e-3).
        A small positive value keeps the operator SPD and the preconditioner
        well-behaved; set smaller to approach a true void.
    """

    def __init__(self, E_solid, nu, penalty=2.0, void_ratio=1e-3):
        self.penalty = float(penalty)
        self.lam_solid, self.mu_solid = lame_from_E_nu(E_solid, nu)
        self.lam_void = self.lam_solid * void_ratio
        self.mu_void = self.mu_solid * void_ratio

    def lame(self, rho):
        """Return (lambda(rho), mu(rho)) as arrays with the shape of ``rho``."""
        f = np.power(rho, self.penalty)
        lam = self.lam_void + f * (self.lam_solid - self.lam_void)
        mu = self.mu_void + f * (self.mu_solid - self.mu_void)
        return lam, mu

    def dlame(self, rho):
        """Return the density derivatives (dlambda/drho, dmu/drho)."""
        df = self.penalty * np.power(rho, self.penalty - 1.0)
        dlam = df * (self.lam_solid - self.lam_void)
        dmu = df * (self.mu_solid - self.mu_void)
        return dlam, dmu

    def d2lame(self, rho):
        """Return the second density derivatives
        (d^2 lambda/drho^2, d^2 mu/drho^2) -- needed for exact
        (second-order-adjoint) Hessian-vector products."""
        p = self.penalty
        d2f = p * (p - 1.0) * np.power(rho, p - 2.0)
        d2lam = d2f * (self.lam_solid - self.lam_void)
        d2mu = d2f * (self.mu_solid - self.mu_void)
        return d2lam, d2mu
