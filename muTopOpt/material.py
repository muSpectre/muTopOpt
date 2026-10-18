#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
SIMP material interpolation for density-based topology optimization.

The design variable is a per-element density ``ρ ∈ [0, 1]``. The isotropic
stiffness is interpolated with the "solid isotropic material with
penalization" (SIMP) power law between a void and a solid phase,

    C(ρ) = C_void + ρ^p (C_solid − C_void).

Both phases are isotropic with the same Poisson's ratio, so this is
equivalent to interpolating the Lamé parameters,

    λ(ρ) = λ_void + ρ^p (λ_solid − λ_void),
    μ(ρ) = μ_void + ρ^p (μ_solid − μ_void),

which is exactly the per-pixel material that
:class:`muGrid.IsotropicStiffnessOperator` consumes (``K_e = 2μG + λV``); no
fourth-order stiffness tensor is stored.

For ``p > 1``, ``ρ^p < ρ`` on ``(0, 1)``: an intermediate density yields less
stiffness than its volume fraction, which penalizes grey material.

The void phase has a small non-zero stiffness ``void_ratio · C_solid``. With
``λ, μ > 0`` in every element, ``K`` is symmetric positive definite on the
periodic fluctuations (modulo rigid translations), so CG is well defined and
the J-FFT (Green-Jacobi) preconditioner applies throughout, including the
diffuse high-contrast interface. The phase contrast ``1/void_ratio`` trades
how closely the void phase approximates true void against the conditioning
of ``K`` and hence the number of CG iterations.
"""

import numpy as np


def lame_from_E_nu(E, nu):
    """Lamé parameters ``(λ, μ)`` of an isotropic material.

    ``λ = Eν / ((1 + ν)(1 − 2ν))``, ``μ = E / (2(1 + ν))``: the plane-strain /
    3D convention, i.e. the Hooke's law ``σ = λ tr(ε) I + 2μ ε`` that the
    stiffness operator assembles.
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def E_nu_from_lame(lam, mu):
    """Young's modulus and Poisson's ratio ``(E, ν)`` from the Lamé parameters.

    The inverse of :func:`lame_from_E_nu`, ``E = μ(3λ + 2μ)/(λ + μ)``,
    ``ν = λ / (2(λ + μ))``, in the same plane-strain / 3D convention (in 2D
    these are the 3D constants of the plane-strain material, not the
    directional in-plane values).
    """
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    return E, nu


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

        .. TODO: rationale for the default ``p = 2``?
    void_ratio : float, optional
        Stiffness ratio of the void phase, ``E_void / E_solid`` (default 1e-3).
        Must be positive for ``K`` to be SPD; smaller values approach a true
        void at the cost of a worse-conditioned ``K`` (see the module
        docstring).

        .. TODO: rationale for the default ``void_ratio = 1e-3``?
    """

    def __init__(self, E_solid, nu, penalty=2.0, void_ratio=1e-3):
        self.penalty = float(penalty)
        self.lam_solid, self.mu_solid = lame_from_E_nu(E_solid, nu)
        self.lam_void = self.lam_solid * void_ratio
        self.mu_void = self.mu_solid * void_ratio

    def lame(self, rho):
        """Return ``(λ(ρ), μ(ρ))`` as arrays with the shape of ``rho``."""
        f = np.power(rho, self.penalty)
        lam = self.lam_void + f * (self.lam_solid - self.lam_void)
        mu = self.mu_void + f * (self.mu_solid - self.mu_void)
        return lam, mu

    def dlame(self, rho):
        """Return the density derivatives ``(dλ/dρ, dμ/dρ)``, i.e.
        ``p ρ^(p−1) (λ_solid − λ_void)`` and likewise for ``μ``."""
        df = self.penalty * np.power(rho, self.penalty - 1.0)
        dlam = df * (self.lam_solid - self.lam_void)
        dmu = df * (self.mu_solid - self.mu_void)
        return dlam, dmu

    def d2lame(self, rho):
        """Return the second density derivatives ``(d²λ/dρ², d²μ/dρ²)``,
        i.e. ``p (p − 1) ρ^(p−2) (λ_solid − λ_void)`` and likewise for ``μ``;
        needed for exact (second-order adjoint) Hessian-vector products."""
        p = self.penalty
        d2f = p * (p - 1.0) * np.power(rho, p - 2.0)
        d2lam = d2f * (self.lam_solid - self.lam_void)
        d2mu = d2f * (self.mu_solid - self.mu_void)
        return d2lam, d2mu
