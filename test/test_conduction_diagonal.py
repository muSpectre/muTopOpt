#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""The Jacobi diagonal of the conduction stiffness ``K = Bᵀ W κ B``.

``diag(K)`` is assembled in closed form, ``K_ii = Σ_o Σ_q W_q κ_{q,i−o}
Σ_α B_{qαo}²``. It must equal ``e_iᵀ K e_i`` for every node on any grid,
including odd extents and odd subdomains under MPI, where neighbouring nodes
across the periodic or rank boundary have the same index parity.
"""

import numpy as np
import pytest

from muTopOpt import SimpConductivity
from muTopOpt.conduction import HomogenizationConductivity


def _probe_diagonal(homog):
    """``e_iᵀ K e_i`` by applying K to one unit vector per global node."""
    u = homog.temperature_field("probe_u")
    Ku = homog.temperature_field("probe_Ku")
    loc = tuple(homog.engine.subdomain_locations)
    n = homog.nb_pixels
    diag = np.zeros(n)
    for g in np.ndindex(*homog.nb_grid_pts):
        u.set_zero()
        local = tuple(gi - li for gi, li in zip(g, loc))
        owned = all(0 <= i < m for i, m in zip(local, n))
        if owned:
            u.p[local] = 1.0
        homog._hessp(u, Ku)
        if owned:
            diag[local] = float(np.asarray(Ku.p)[local])
    return diag


@pytest.mark.parametrize("nb_grid_pts", [(5, 3), (4, 6), (3, 5, 3)])
@pytest.mark.parametrize("element", ["q1", "p1"])
def test_diagonal_is_exact(comm, nb_grid_pts, element):
    material = SimpConductivity(1.0, penalty=3.0, void_ratio=1e-2)
    homog = HomogenizationConductivity(
        nb_grid_pts, material, comm=comm, element=element,
        domain_lengths=[0.5 * n for n in nb_grid_pts][::-1],
        preconditioner="green")
    rng = np.random.default_rng(3)
    rho_global = rng.random(nb_grid_pts)
    loc = tuple(homog.engine.subdomain_locations)
    owned = tuple(slice(l, l + m) for l, m in zip(loc, homog.nb_pixels))
    homog.set_density(rho_global[owned])

    diag = homog.temperature_field("diag")
    homog._assemble_diagonal(diag)
    np.testing.assert_allclose(np.asarray(diag.p), _probe_diagonal(homog),
                               rtol=1e-12, atol=0)


def test_device_matches_host(comm):
    """Objective and gradient on the accelerator equal the host result to
    rounding (same discrete operator, same solves)."""
    import muGrid

    if not muGrid.has_gpu or not muGrid.is_gpu_available():
        pytest.skip("no GPU")
    pytest.importorskip("cupy")
    from muTopOpt.conduction_problem import FluxTargetProblem
    from muTopOpt.loadcases_conduction import (
        isotropic_conductivity_tensor,
        target_load_cases,
    )

    rho = np.random.default_rng(1).random((24, 20))
    result = {}
    for device in ("cpu", "gpu"):
        homog = HomogenizationConductivity(
            (24, 20), SimpConductivity(1.0, 3.0, 1e-3), comm=comm,
            device=device, cg_tol=1e-10)
        if comm.size > 1:
            pytest.skip("serial comparison")
        cases = target_load_cases(2, isotropic_conductivity_tensor(2, 0.3))
        f, g = FluxTargetProblem(homog, cases).objective_and_gradient(rho)
        result[device] = (f, np.asarray(g))
    assert np.isclose(result["cpu"][0], result["gpu"][0], rtol=1e-12)
    np.testing.assert_allclose(result["gpu"][1], result["cpu"][1],
                               rtol=0, atol=1e-9 * np.abs(result["cpu"][1]).max())
