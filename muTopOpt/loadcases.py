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


def energy_load_cases(dim, target_sigma, magnitude=1.0, weight=1.0):
    """Load cases of the *energy* objective: every independent entry of the
    effective stiffness is matched relative to its own target.

    The effective stiffness has ``n(n+1)/2`` independent entries for
    ``n = dim(dim+1)/2`` strain directions; an isotropic target constrains
    9 of them in 3D (4 in 2D): the three normal stiffnesses, the three
    couplings between normal strains and the three shears. They are the
    energies ``W = E_k : C̄ : Ē^Γ`` of a load-case strain ``Ē^Γ`` with a
    "left" strain ``E_k``:

    * normal case ``i``: left strains of the normal directions ``j ≥ i``
      (``C̄_iijj``, each coupling once);
    * shear case: its own strain (``C̄_ijij``).

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    target_sigma : callable
        ``target_sigma(E) -> σ`` of the target stiffness.
    magnitude : float, optional
        Magnitude of the unit strains (default 1).
    weight : float, optional
        Weight ``a_Γ`` of every case (default 1).

    Returns
    -------
    list of LoadCase
        One per unit strain, in the order of :func:`unit_strains`, each
        with its ``energy_terms``.
    """
    strains = unit_strains(dim, magnitude)
    cases = []
    for index, strain in enumerate(strains):
        target = target_sigma(strain)
        lefts = range(index, dim) if index < dim else [index]
        terms = [(strains[k], float(np.sum(strains[k] * target)), 1.0)
                 for k in lefts]
        cases.append(LoadCase(strain, target, weight, terms))
    return cases


def auxetic_target(dim, E_solid, nu=-0.5, scale=1.0, shear_fraction=0.15):
    """Isotropic target with a negative Poisson ratio, as a callable
    ``sigma(E)``.

    The target is fixed by its Poisson ratio ``nu`` and its shear modulus
    ``G* = scale · shear_fraction · E_solid``. The shear modulus sets how
    stiff the metamaterial is asked to be relative to the solid it is made
    of: a porous cell cannot reach the solid's own stiffness, and a target
    that is too stiff leaves no room for the voids an auxetic mechanism
    needs. ``scale`` rescales the whole target stiffness at a fixed Poisson
    ratio.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    E_solid : float
        Young's modulus of the solid phase.
    nu : float, optional
        Target Poisson ratio (default -0.5).
    scale : float, optional
        Factor on the target stiffness (default 1).
    shear_fraction : float, optional
        Target shear modulus at ``scale = 1`` as a fraction of ``E_solid``
        (default 0.15).
    """
    shear = scale * shear_fraction * E_solid
    return isotropic_stiffness_from_E_nu(dim, 2.0 * shear * (1.0 + nu), nu)
