#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Effective (homogenized) elastic properties of a density design.

The effective stiffness ``C̄`` is measured column by column: for each
independent unit macro strain ``Ē^k`` (see
:func:`muTopOpt.loadcases.unit_strains`) the fluctuation ``u`` solves
``K u = −Bᵀ C:Ē^k`` and the homogenized stress
``σ̄^k = (1/V) ∫ C:(Ē^k + ∇ˢu) dV`` is the ``k``-th column of ``C̄`` in Voigt
notation (3×3 in 2D, 6×6 in 3D, engineering shear convention). The effective
bulk/shear moduli and Poisson's ratio follow from ``C̄``.
"""

import numpy as np

from .loadcases import unit_strains


def _to_voigt_stress(sigma, dim):
    """Symmetric ``dim × dim`` stress -> Voigt vector (normal components,
    then ``yz, xz, xy`` in 3D or ``xy`` in 2D)."""
    if dim == 2:
        return np.array([sigma[0, 0], sigma[1, 1], sigma[0, 1]])
    return np.array([
        sigma[0, 0], sigma[1, 1], sigma[2, 2],
        sigma[1, 2], sigma[0, 2], sigma[0, 1],
    ])


def effective_stiffness(homogenization, rho):
    """Effective stiffness ``C̄`` (Voigt) of the density ``rho``.

    Column ``k`` is the Voigt stress response to the ``k``-th unit strain
    (diagonal strains first, then unit engineering shears), so ``C̄`` maps a
    Voigt strain (engineering shears) to a Voigt stress.

    Parameters
    ----------
    homogenization : Homogenization
        Solver whose material is set to ``rho`` (a side effect of this call).
    rho : array_like
        Element-wise density of shape :attr:`Homogenization.nb_pixels`.

    Returns
    -------
    numpy.ndarray
        ``C̄`` of shape ``(3, 3)`` in 2D or ``(6, 6)`` in 3D.
    """
    h = homogenization
    h.set_density(np.asarray(rho))
    strains = unit_strains(h.dim, magnitude=1.0)
    u = h.vector_field("to_Ceff_u")
    cols = []
    for E in strains:
        h.solve_macro(E, u)
        sigma = h.homogenized_stress(u, E)
        cols.append(_to_voigt_stress(sigma, h.dim))
    return np.array(cols).T  # C_eff[:, k] = response to unit strain k


def isotropic_moduli_2d(C):
    """Return ``(K, G, poisson, E, zener)`` from a 2D Voigt stiffness ``C``
    (3×3).

    ``K = (C11 + C12)/2`` is the plane-strain area (bulk) modulus,
    ``G = C33`` the shear modulus,
    ``poisson`` the directional Poisson ratio ``-S12/S11`` from the compliance,
    ``E`` the directional Young's modulus ``1/S11``, and ``zener`` the anisotropy
    ratio ``2 C33 / (C11 - C12)`` (1 for isotropic)."""
    C = np.asarray(C)
    K = 0.5 * (C[0, 0] + C[0, 1])
    G = C[2, 2]
    S = np.linalg.inv(C)
    poisson = -S[0, 1] / S[0, 0]
    E = 1.0 / S[0, 0]
    zener = 2.0 * C[2, 2] / (C[0, 0] - C[0, 1])
    return K, G, poisson, E, zener
