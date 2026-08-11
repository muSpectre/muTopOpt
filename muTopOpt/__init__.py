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

    material = SimpMaterial(E_solid=1.0, nu=0.3)
    homog = Homogenization((64, 64), material)
    cases = target_load_cases(2, isotropic_stiffness_tensor(2, K=0.1, G=0.05))
    reg = PhaseFieldRegularization(homog)  # eta defaults to one grid spacing
    problem = StressTargetProblem(homog, cases, regularization=reg)
    rho, info = optimize_bounded_lbfgs(problem, initial_density(homog.nb_pixels))

Heat conduction (scalar diffusion) is available alongside elasticity, with
the same optimizer drivers and phase-field regularization::

    from muTopOpt import SimpConductivity, HomogenizationConductivity, FluxTargetProblem
    from muTopOpt.loadcases_conduction import (
        isotropic_conductivity_tensor, target_load_cases as target_flux_cases)
    from muTopOpt.optimize import initial_density, optimize_bounded_lbfgs

    material = SimpConductivity(kappa_solid=1.0)
    homog = HomogenizationConductivity((64, 64), material)
    cases = target_flux_cases(2, isotropic_conductivity_tensor(2, kappa=0.1))
    reg = PhaseFieldRegularization(homog)
    problem = FluxTargetProblem(homog, cases, regularization=reg)
    rho, info = optimize_bounded_lbfgs(problem, initial_density(homog.nb_pixels))
"""

__version__ = "0.0.1"

from .conduction import HomogenizationConductivity, SimpConductivity
from .conduction_problem import FluxLoadCase, FluxTargetProblem
from .homogenization import Homogenization
from .material import SimpMaterial, E_nu_from_lame, lame_from_E_nu
from .nodal import ConsistentDoubleWell, NodalElementMap
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
    "E_nu_from_lame",
    "LoadCase",
    "StressTargetProblem",
    "HomogenizationConductivity",
    "SimpConductivity",
    "FluxLoadCase",
    "FluxTargetProblem",
    "PhaseFieldRegularization",
    "NodalPhaseFieldRegularization",
    "NodalElementMap",
    "ConsistentDoubleWell",
    "fe_laplacian_stencil",
    "effective_stiffness",
    "isotropic_moduli_2d",
]
