#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Helpers to build the independent load cases that constrain an effective
stiffness, and to translate a target effective stiffness into per-load-case
target stresses.

In ``dim`` dimensions the symmetric strain space has ``dim*(dim+1)/2``
independent directions (3 in 2D, 6 in 3D). Prescribing one unit macro strain per
direction and the stress it should produce fully constrains the effective
(isotropic or anisotropic) stiffness -- the setup used to design metamaterials
for a target bulk/shear modulus or Poisson's ratio.
"""

import numpy as np

from .problem import LoadCase


def _voigt_directions(dim):
    """Independent symmetric-strain directions as (i, j) index pairs: the
    diagonal terms first, then the shears."""
    diag = [(i, i) for i in range(dim)]
    shear = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    return diag + shear


def unit_strains(dim, magnitude=1.0):
    """The ``dim*(dim+1)/2`` unit macro strains (symmetric, dim x dim)."""
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
    """Fourth-order isotropic stiffness as a function acting on a strain tensor:
    ``σ = 2G ε_dev + dim*K ε_vol``... returned as a callable ``sigma(E)``."""
    def sigma(E):
        E = np.asarray(E)
        tr = np.trace(E)
        dev = E - tr / dim * np.eye(dim)
        return 2.0 * G * dev + K * tr * np.eye(dim)

    return sigma


def target_load_cases(dim, target_sigma, magnitude=1.0, weights=None):
    """Build load cases from a callable ``target_sigma(E) -> σ`` (e.g.
    :func:`isotropic_stiffness_tensor`) evaluated on the unit strains."""
    strains = unit_strains(dim, magnitude)
    if weights is None:
        weights = [1.0] * len(strains)
    return [
        LoadCase(E, target_sigma(E), w)
        for E, w in zip(strains, weights)
    ]
