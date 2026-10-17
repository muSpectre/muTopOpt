"""Single-precision (float32) solves.

When the grid fields are single precision the whole forward/adjoint path -- the
fields, the preconditioner (complex64 FFT buffers, float32 kernel/diagonal) and
the CG scratch -- runs in single precision, with only the scalar accumulators
(objective, reductions, the host-side L-BFGS design variables) in double
precision. muGrid threads the field precision through its preconditioner
factories, so ``Homogenization(..., dtype=np.float32)`` is all that is needed.
"""

import numpy as np
import pytest

from muTopOpt import (
    Homogenization,
    PhaseFieldRegularization,
    SimpMaterial,
    StressTargetProblem,
)
from muTopOpt.loadcases import isotropic_stiffness_tensor, target_load_cases


def _rho(h):
    return np.random.default_rng(0).uniform(0.2, 0.8, h.nb_pixels)


def _assert_fields_single(h, p):
    for fld in (h.lam, h.mu, h._rhs, h._Ku, p._u, p._adj, p._adj_rhs):
        assert np.dtype(fld.dtype) == np.dtype(np.float32)


@pytest.mark.parametrize("preconditioner", ["green-jacobi", "green"])
def test_single_precision_solve(comm, preconditioner):
    mat = SimpMaterial(1.0, 0.3, 3.0, 1e-3)
    h = Homogenization((16, 16), mat, comm=comm, cg_tol=1e-4,
                       preconditioner=preconditioner, dtype=np.float32)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.08, G=0.03), magnitude=0.01)
    p = StressTargetProblem(h, cases, regularization=PhaseFieldRegularization(h))
    f, g = p.objective_and_gradient(_rho(h))

    # All grid fields stayed single precision. That the solve *completed* with
    # float32 fields is itself proof the preconditioner is single precision: a
    # double-precision (complex128) FFT buffer would clash with the float32
    # residual at engine.fft() and raise.
    _assert_fields_single(h, p)
    # Accumulators stay double precision (objective scalar, host-side gradient).
    assert isinstance(f, float)
    assert np.asarray(g).dtype == np.dtype(np.float64)


def test_green_jacobi_buffers_are_single_precision(comm):
    mat = SimpMaterial(1.0, 0.3, 3.0, 1e-3)
    h = Homogenization((16, 16), mat, comm=comm, dtype=np.float32)
    h.set_density(np.full(h.nb_pixels, 0.5))
    # The preconditioner's FFT work buffer is complex64 (single precision).
    assert np.dtype(h._prec._green._work.dtype) == np.dtype(np.complex64)


def test_single_matches_double(comm):
    # A single-precision homogenized stress agrees with the double-precision one
    # to single-precision accuracy (correctness, not just "it runs").
    mat = SimpMaterial(1.0, 0.3, 3.0, 1e-3)
    cases = target_load_cases(
        2, isotropic_stiffness_tensor(2, K=0.08, G=0.03), magnitude=0.01)
    rho = np.random.default_rng(1).uniform(0.2, 0.8, (16, 16))
    stresses = {}
    for dt in (np.float64, np.float32):
        h = Homogenization((16, 16), mat, comm=comm, cg_tol=1e-5, dtype=dt)
        h.set_density(rho)
        u = h.vector_field("probe_u")
        h.solve_macro(cases[0].macro_strain, u, rtol=1e-5)
        stresses[dt] = h.homogenized_stress(u, cases[0].macro_strain)
    assert np.allclose(stresses[np.float32], stresses[np.float64],
                       rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("nb_grid_pts", [(64, 64), (128, 128)])
def test_float32_field_reduction_does_not_degrade_with_size(nb_grid_pts):
    """The solve's norms must come from muGrid's double-accumulating
    reductions, not from BLAS ``sdot`` on the raw float32 buffer.

    ``xp.dot`` on a float32 array accumulates in float32, so its error grows
    *linearly* in the number of entries: measured 4.2e-9 at 32^3 but 5.5e-6 at
    128^3, and ~4.6e-4 at 512^3 -- the only quantity in the whole solve whose
    accuracy degrades with resolution this way. ``linalg.norm_sq`` accumulates
    (and returns) in double and stays at ~1e-14 throughout. It also avoids the
    full-size contiguous copy that ``.p.ravel()`` makes whenever the
    collection carries ghosts.

    This test pins the *relationship*, not an absolute error: whatever
    ``solve_rhs`` uses for ``|b|`` must track the double-precision reference
    far more closely than the naive float32 dot does.
    """
    import muGrid
    from muGrid import linalg

    ghosts = (1,) * len(nb_grid_pts)
    fc = muGrid.GlobalFieldCollection(
        nb_grid_pts, nb_ghosts_left=ghosts, nb_ghosts_right=ghosts
    )
    f = fc.real_field("f", (len(nb_grid_pts),), dtype=np.float32)
    rng = np.random.default_rng(0)
    f.p[...] = rng.standard_normal(f.p.shape).astype(np.float32)

    host = np.asarray(f.p).astype(np.float64).ravel()
    exact = float(np.dot(host, host))

    flat = f.p.ravel()
    naive_err = abs(float(np.dot(flat, flat)) - exact) / exact
    reduced_err = abs(float(linalg.norm_sq(f)) - exact) / exact

    assert reduced_err < 1e-12, f"norm_sq relative error {reduced_err:.2e}"
    # Comfortably true in practice (7+ orders at 128^2); assert the margin
    # loosely so the test is about the mechanism, not the RNG draw.
    assert reduced_err < naive_err / 100.0, (
        f"norm_sq {reduced_err:.2e} vs sdot {naive_err:.2e}"
    )
