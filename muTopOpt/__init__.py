#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
muTopOpt -- FFT-accelerated FE topology optimization of mechanical metamaterials.

Element-wise (per-pixel) density formulation with phase-field regularization and
a stress-matching objective, built on the fused, GPU-capable operators and the
J-FFT (Green-Jacobi) preconditioner of muGrid. Dimension-agnostic: the same code
runs 2D and 3D unit cells.

Typical use::

    from muTopOpt import (SimpMaterial, Homogenization, StressTargetProblem,
                          PhaseFieldRegularization)
    from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases
    from muTopOpt.optimize import initial_density, optimize_bounded_lbfgs

    material = SimpMaterial(E_solid=1.0, nu=0.3, penalty=3.0)
    homog = Homogenization((64, 64), material)
    cases = target_load_cases(2, isotropic_stiffness_tensor(2, K=0.1, G=0.05))
    reg = PhaseFieldRegularization(homog, eta=1.0, well_weight=1e-3)
    problem = StressTargetProblem(homog, cases, regularization=reg)
    rho, info = optimize_bounded_lbfgs(problem, initial_density(homog.nb_pixels))
"""

__version__ = "0.0.1"

from .homogenization import Homogenization
from .material import SimpMaterial, lame_from_E_nu
from .problem import LoadCase, StressTargetProblem
from .properties import effective_stiffness, isotropic_moduli_2d
from .regularization import (
    NodalPhaseFieldRegularization,
    PhaseFieldRegularization,
    fe_laplacian_stencil,
)

__all__ = [
    "Homogenization",
    "SimpMaterial",
    "lame_from_E_nu",
    "LoadCase",
    "StressTargetProblem",
    "PhaseFieldRegularization",
    "NodalPhaseFieldRegularization",
    "fe_laplacian_stencil",
    "effective_stiffness",
    "isotropic_moduli_2d",
]
