#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Load cases that constrain an effective stiffness, and the translation of a
target effective stiffness into per-load-case target stresses.

In ``d`` dimensions the symmetric strain space has ``d(d+1)/2`` independent
directions (3 in 2D, 6 in 3D). Prescribing one unit macro strain ``Ē^Γ`` per
direction together with the stress ``σ*^Γ = C*:Ē^Γ`` of the target stiffness
``C*`` fixes every column of the effective stiffness ``C̄`` (isotropic or
anisotropic) -- the setup used to design metamaterials for a target
bulk/shear modulus or Poisson's ratio.

Isotropic targets use the convention ``σ = 2G dev(ε) + K tr(ε) I``, i.e.
``G = μ`` and ``K = λ + 2μ/d``: the ordinary bulk modulus in 3D and the
plane-strain area modulus in 2D.
"""

import numpy as np

from .material import lame_from_E_nu
from .problem import LoadCase


def _voigt_directions(dim):
    """Independent symmetric-strain directions as ``(i, j)`` index pairs: the
    diagonal terms first, then the shears ``i < j`` in row-major order."""
    diag = [(i, i) for i in range(dim)]
    shear = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    return diag + shear


def unit_strains(dim, magnitude=1.0):
    """The ``dim(dim+1)/2`` unit macro strains as symmetric ``dim × dim``
    arrays, ordered as in :func:`_voigt_directions`.

    A diagonal case has ``E_ii = magnitude``; a shear case has
    ``E_ij = E_ji = magnitude/2``, i.e. an engineering shear strain
    ``γ_ij = magnitude``.
    """
    strains = []
    for (i, j) in _voigt_directions(dim):
        E = np.zeros((dim, dim))
        if i == j:
            E[i, i] = magnitude
        else:
            E[i, j] = E[j, i] = 0.5 * magnitude
        strains.append(E)
    return strains


def isotropic_stiffness_tensor(dim, K, G):
    """Isotropic stiffness ``C*`` with bulk modulus ``K`` and shear modulus
    ``G``, as a callable ``sigma(E)`` acting on a ``dim × dim`` strain:

        σ(E) = 2G (E − tr(E)/dim · I) + K tr(E) I.

    ``K = λ + 2μ/dim`` (3D bulk modulus, 2D plane-strain area modulus) and
    ``G = μ``.
    """
    def sigma(E):
        E = np.asarray(E)
        tr = np.trace(E)
        dev = E - tr / dim * np.eye(dim)
        return 2.0 * G * dev + K * tr * np.eye(dim)

    return sigma


def isotropic_stiffness_from_E_nu(dim, E, nu):
    """Isotropic stiffness from Young's modulus ``E`` and Poisson's ratio
    ``nu``, in the plane-strain / 3D Hooke's-law convention of the mechanics
    operator (the same convention as :class:`~muTopOpt.material.SimpMaterial`):
    ``σ = λ tr(ε) I + 2μ ε``. Returned as a callable ``sigma(E_macro)``,
    interchangeable with :func:`isotropic_stiffness_tensor`.

    A target built from the *solid* ``(E, nu)`` is exactly the response of a
    fully dense (``ρ = 1``) cell."""
    lam, mu = lame_from_E_nu(E, nu)

    def sigma(E_macro):
        E_macro = np.asarray(E_macro)
        return lam * np.trace(E_macro) * np.eye(dim) + 2.0 * mu * E_macro

    return sigma


def target_load_cases(dim, target_sigma, magnitude=1.0, weights=None):
    """Build one :class:`~muTopOpt.problem.LoadCase` per unit strain.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    target_sigma : callable
        ``target_sigma(E) -> σ``, the target stress for a ``dim × dim``
        strain (e.g. :func:`isotropic_stiffness_tensor`).
    magnitude : float, optional
        Magnitude of the unit strains (default 1, see :func:`unit_strains`).
    weights : sequence of float, optional
        Per-case weights ``a_Γ`` of the objective (default all 1).

    Returns
    -------
    list of LoadCase
        ``dim(dim+1)/2`` cases, diagonal strains first, then shears.
    """
    strains = unit_strains(dim, magnitude)
    if weights is None:
        weights = [1.0] * len(strains)
    return [
        LoadCase(E, target_sigma(E), w)
        for E, w in zip(strains, weights)
    ]
