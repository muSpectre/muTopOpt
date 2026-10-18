#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
muTopOpt -- FFT-accelerated FE topology optimization of mechanical metamaterials.

Designs a periodic unit cell whose effective stiffness matches a target. The
design variable is a density ``ρ ∈ [0, 1]``, either per element (pixel) or
per node; the material is a SIMP interpolation of the Lamé parameters, the
objective a stress misfit over macro-strain load cases plus a Modica-Mortola
phase-field regularization, and the gradient comes from the discrete adjoint
method. Forward and adjoint problems are solved with the fused, GPU-capable
stiffness operator and the J-FFT (Green-Jacobi) preconditioned CG of muGrid.
Dimension-agnostic: the same code runs 2D and 3D unit cells.

The theory is documented where it is implemented:

* :mod:`muTopOpt.material` -- SIMP interpolation;
* :mod:`muTopOpt.homogenization` -- the cell problem ``K u = −Bᵀ C:Ē``, the
  homogenized stress, CG safeguards and GPU memory policy;
* :mod:`muTopOpt.loadcases` -- macro-strain load cases and isotropic targets;
* :mod:`muTopOpt.problem` -- the stress-matching objective, its adjoint
  gradient, the adjoint-corrected Lagrangian and Hessian-vector products;
* :mod:`muTopOpt.regularization` -- the Modica-Mortola functional;
* :mod:`muTopOpt.nodal` -- nodal design fields and exact double-well
  integration;
* :mod:`muTopOpt.bounds` -- Voigt and Hashin-Shtrikman feasibility bounds.

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

__version__ = "1.1.0"

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
from .timing import timer
from .version import git_revision, mugrid_version, mutopopt_version, version_string

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
    "timer",
    "git_revision",
    "mugrid_version",
    "mutopopt_version",
    "version_string",
]
