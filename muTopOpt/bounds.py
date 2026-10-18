#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Back-of-the-envelope feasibility bounds for a target effective stiffness.

A porous (solid + void) microstructure at solid volume fraction ``phi`` cannot
exceed the Voigt (rule-of-mixtures) bound and, more sharply, the
Hashin-Shtrikman upper bound on its effective bulk and shear moduli. The
corresponding lower bounds (Reuss, Hashin-Shtrikman lower) vanish for a true
void phase, so only the upper bounds constrain feasibility: inverting them in
``phi`` gives the *minimum* solid fraction any microstructure needs to reach a
target modulus.

Conventions match the rest of muTopOpt: ``G`` is the Lamé shear modulus ``μ``
and ``K = λ + 2μ/d`` -- the ordinary bulk modulus in 3D, the plane-strain
area modulus in 2D. The 2D bounds are the transverse (plane-strain) Hashin
bounds in exactly this parameterization.

For a solid of moduli ``(K₀, G₀)`` at volume fraction ``φ`` and a void of
zero stiffness, the Voigt bound is ``(φK₀, φG₀)`` and the Hashin-Shtrikman
upper bound is

    K_HS = K₀ + (1 − φ) / (−1/K₀ + φ/(K₀ + G₀)),             (d = 2)
    K_HS = K₀ + (1 − φ) / (−1/K₀ + 3φ/(3K₀ + 4G₀)),          (d = 3)
    G_HS = G₀ + (1 − φ) / (−1/G₀ + φ/(G₀ + ζ)),

with ``ζ = K₀G₀/(K₀ + 2G₀)`` in 2D and
``ζ = G₀(9K₀ + 8G₀)/(6(K₀ + 2G₀))`` in 3D. Both bounds increase
monotonically in ``φ``, so the minimum solid fraction for a target modulus
is found by bisection.

These bound each modulus *separately*; cross-property bounds coupling K and G
are sharper still, so a target that passes this check can still be hard (or
impossible) to reach. It is an envelope check, not a certificate.
"""

import numpy as np

from .material import lame_from_E_nu


def solid_moduli(dim, E, nu):
    """``(K, G)`` of an isotropic material given Young's modulus and Poisson's
    ratio, in muTopOpt's convention ``K = λ + 2μ/dim`` (3D bulk modulus / 2D
    plane-strain area modulus) and ``G = μ``."""
    lam, mu = lame_from_E_nu(E, nu)
    return lam + 2.0 * mu / dim, mu


def voigt_upper(dim, phi, K0, G0):
    """Voigt (rule-of-mixtures) upper bound (K, G) of a porous solid at solid
    volume fraction ``phi``."""
    return phi * K0, phi * G0


def hashin_shtrikman_upper(dim, phi, K0, G0):
    """Hashin-Shtrikman upper bound (K, G) of a porous solid (void phase of
    zero stiffness) at solid volume fraction ``phi``."""
    if dim == 2:
        zeta = K0 * G0 / (K0 + 2.0 * G0)
        K = K0 + (1.0 - phi) / (-1.0 / K0 + phi / (K0 + G0))
        G = G0 + (1.0 - phi) / (-1.0 / G0 + phi / (G0 + zeta))
    else:
        zeta = G0 * (9.0 * K0 + 8.0 * G0) / (6.0 * (K0 + 2.0 * G0))
        K = K0 + (1.0 - phi) / (-1.0 / K0 + 3.0 * phi / (3.0 * K0 + 4.0 * G0))
        G = G0 + (1.0 - phi) / (-1.0 / G0 + phi / (G0 + zeta))
    return K, G


def _minimum_phi(bound, target):
    """Smallest ``phi`` in [0, 1] with ``bound(phi) >= target`` (80 bisection
    steps on the monotonically increasing ``bound``, i.e. to double-precision
    resolution); 0 for a non-positive target, ``inf`` if even the solid
    (``phi = 1``) falls short."""
    if target <= 0.0:
        return 0.0
    if target > bound(1.0):
        return np.inf
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bound(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def target_feasibility(dim, target_K, target_G, solid_E, solid_nu,
                       phi_low=0.2, phi_high=0.8):
    """Envelope feasibility check of a target ``(K, G)`` against the solid.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    target_K, target_G : float
        Target moduli, in the convention of :func:`solid_moduli`.
    solid_E, solid_nu : float
        Young's modulus and Poisson's ratio of the solid phase.
    phi_low, phi_high : float, optional
        Solid fractions below / above which a warning is issued (defaults
        0.2 and 0.8).

    Returns
    -------
    dict
        The solid moduli (``K0``, ``G0``); the minimum solid fraction each
        target modulus requires under the Hashin-Shtrikman upper bound
        (``phi_hs_K``, ``phi_hs_G``, ``inf`` if unreachable) and under the
        looser Voigt bound (``phi_voigt``, the larger of the K and G
        requirements); ``phi_min = max(phi_hs_K, phi_hs_G)``; and a list of
        human-readable ``warnings``: when a target modulus exceeds the
        solid's, when reaching the target needs a solid fraction above
        ``phi_high`` (near-solid designs), or when bound-efficient designs
        would sit below ``phi_low`` (sparse structures that may fail to
        percolate).
    """
    K0, G0 = solid_moduli(dim, solid_E, solid_nu)
    phi_hs_K = _minimum_phi(
        lambda p: hashin_shtrikman_upper(dim, p, K0, G0)[0], target_K)
    phi_hs_G = _minimum_phi(
        lambda p: hashin_shtrikman_upper(dim, p, K0, G0)[1], target_G)
    phi_voigt = max(target_K / K0 if target_K <= K0 else np.inf,
                    target_G / G0 if target_G <= G0 else np.inf)
    phi_min = max(phi_hs_K, phi_hs_G)

    warnings = []
    if target_K > K0:
        warnings.append(
            f"target K={target_K:.4g} exceeds the solid's K={K0:.4g}: "
            "unreachable at any density")
    if target_G > G0:
        warnings.append(
            f"target G={target_G:.4g} exceeds the solid's G={G0:.4g}: "
            "unreachable at any density")
    if not warnings:
        if phi_min > phi_high:
            binding = "G" if phi_hs_G >= phi_hs_K else "K"
            warnings.append(
                f"target {binding} requires a solid fraction of at least "
                f"{phi_min:.0%} (Hashin-Shtrikman bound; Voigt: "
                f"{phi_voigt:.0%}): expect near-solid designs with little "
                "room left to tune the other modulus")
        elif phi_min < phi_low:
            warnings.append(
                f"the target moduli are reachable from a solid fraction of "
                f"{phi_min:.0%} on (Hashin-Shtrikman bound): bound-efficient "
                "designs will be very sparse and may fail to percolate")

    return {
        "K0": K0,
        "G0": G0,
        "phi_hs_K": phi_hs_K,
        "phi_hs_G": phi_hs_G,
        "phi_voigt": phi_voigt,
        "phi_min": phi_min,
        "warnings": warnings,
    }
