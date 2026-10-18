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
effective (possibly anisotropic) conductivity tensor: for ``Ē = m e_i`` the
homogenized flux is ``q̄ = m κ_eff e_i``, the ``i``-th column of ``κ_eff``
scaled by ``m``. Fluxes follow the constitutive sign convention
``q = κ ∇T`` of :mod:`muTopOpt.conduction`.
"""

import numpy as np

from .conduction_problem import FluxLoadCase


def unit_gradients(dim, magnitude=1.0):
    """The ``dim`` macro temperature gradients ``magnitude · e_i`` (one per
    axis), as a list of length-``dim`` arrays."""
    grads = []
    for i in range(dim):
        E = np.zeros(dim)
        E[i] = magnitude
        grads.append(E)
    return grads


def isotropic_conductivity_tensor(dim, kappa):
    """Isotropic conductivity as a callable ``q(E) = kappa * E`` acting on a
    gradient vector. ``dim`` is not used (the signature matches
    :func:`muTopOpt.loadcases.isotropic_stiffness_tensor`)."""
    def q(E):
        return kappa * np.asarray(E, dtype=float)

    return q


def target_load_cases(dim, target_flux, magnitude=1.0, weights=None):
    """Build load cases from a callable ``target_flux(E) -> q`` (e.g.
    :func:`isotropic_conductivity_tensor`, or ``lambda E: kappa_target @ E``
    for a general anisotropic target tensor) evaluated on the gradients
    :func:`unit_gradients` ``(dim, magnitude)``.

    ``weights`` gives one weight per load case (default 1 each). Returns a
    list of ``dim`` :class:`~muTopOpt.conduction_problem.FluxLoadCase`."""
    grads = unit_gradients(dim, magnitude)
    if weights is None:
        weights = [1.0] * len(grads)
    return [
        FluxLoadCase(E, target_flux(E), w)
        for E, w in zip(grads, weights)
    ]
