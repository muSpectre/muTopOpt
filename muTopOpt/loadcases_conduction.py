#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Helpers to build the independent load cases that constrain an effective
conductivity, and to translate a target effective conductivity tensor into
per-load-case target fluxes.

Unlike elasticity's symmetric strain (``dim*(dim+1)/2`` independent
directions), a macroscopic temperature gradient is just a ``dim``-vector, so
``dim`` independent unit gradients (one per axis) already fully constrain the
effective (possibly anisotropic) conductivity tensor.
"""

import numpy as np

from .conduction_problem import FluxLoadCase


def unit_gradients(dim, magnitude=1.0):
    """The ``dim`` unit macro temperature gradients (one per axis)."""
    grads = []
    for i in range(dim):
        E = np.zeros(dim)
        E[i] = magnitude
        grads.append(E)
    return grads


def isotropic_conductivity_tensor(dim, kappa):
    """Isotropic conductivity as a function acting on a gradient vector:
    ``q = kappa * E``... returned as a callable ``q(E)``."""
    def q(E):
        return kappa * np.asarray(E, dtype=float)

    return q


def target_load_cases(dim, target_flux, magnitude=1.0, weights=None):
    """Build load cases from a callable ``target_flux(E) -> q`` (e.g.
    :func:`isotropic_conductivity_tensor`, or ``lambda E: kappa_target @ E``
    for a general anisotropic target tensor) evaluated on the unit
    gradients."""
    grads = unit_gradients(dim, magnitude)
    if weights is None:
        weights = [1.0] * len(grads)
    return [
        FluxLoadCase(E, target_flux(E), w)
        for E, w in zip(grads, weights)
    ]
