#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
FFT-accelerated FE micromechanical homogenization on a regular grid.

:class:`Homogenization` owns the muGrid machinery -- the FFT engine (which
doubles as the ghosted domain decomposition), the fused
``IsotropicStiffnessOperator`` (the system matrix ``K``), the per-pixel Lamé
fields, and the J-FFT (Green-Jacobi) preconditioner -- and exposes the three
operations the optimizer needs:

* :meth:`solve_macro`   -- solve ``K(rho) u = -Bᵀ C:Ē`` for a macro strain ``Ē``;
* :meth:`solve_rhs`     -- solve ``K(rho) x = b`` for an arbitrary rhs (adjoint);
* :meth:`homogenized_stress` -- the cell-averaged stress ``⟨σ⟩``.

Everything is dimension-agnostic (``dim`` = 2 or 3): the only per-dimension
detail is which ``IsotropicStiffnessOperator{2,3}D`` to instantiate. Fields live
on the engine's real-space collection so the solver, the preconditioner FFTs and
the operators all share one ghosted, MPI-decomposed, (optionally) device-resident
layout.
"""

import numpy as np

import muGrid
from muGrid.Preconditioners import (
    make_green_jacobi_preconditioner,
    make_reference_stiffness_preconditioner,
)
from muGrid.Solvers import conjugate_gradients

from .material import SimpMaterial


class Homogenization:
    def __init__(
        self,
        nb_grid_pts,
        material: SimpMaterial,
        comm=None,
        domain_lengths=None,
        element="q1",
        preconditioner="green-jacobi",
        cg_tol=1e-8,
        cg_maxiter=2000,
    ):
        self.dim = len(nb_grid_pts)
        if self.dim not in (2, 3):
            raise ValueError("nb_grid_pts must be 2- or 3-dimensional")
        self.nb_grid_pts = tuple(int(n) for n in nb_grid_pts)
        self.material = material
        self.comm = comm if comm is not None else muGrid.Communicator()

        if domain_lengths is None:
            domain_lengths = [1.0] * self.dim
        self.domain_lengths = [float(x) for x in domain_lengths]
        self.grid_spacing = [
            L / n for L, n in zip(self.domain_lengths, self.nb_grid_pts)
        ]
        self.domain_volume = float(np.prod(self.domain_lengths))

        ghosts = (1,) * self.dim
        self.engine = muGrid.FFTEngine(
            self.nb_grid_pts, self.comm,
            nb_ghosts_left=ghosts, nb_ghosts_right=ghosts,
        )
        self.fc = self.engine.real_space_collection

        elem = (
            muGrid.FEMElement.q1 if element == "q1" else muGrid.FEMElement.p1
        )
        self.element = elem  # muGrid.FEMElement, shared with the regularization
        self.element_name = element
        OpCls = (
            muGrid.IsotropicStiffnessOperator2D
            if self.dim == 2
            else muGrid.IsotropicStiffnessOperator3D
        )
        self.op = OpCls(self.grid_spacing, elem)

        # Per-pixel material (scalar fields) and solver scratch.
        self.lam = self.fc.real_field("to_lambda")
        self.mu = self.fc.real_field("to_mu")
        self._rhs = self.fc.real_field("to_rhs", (self.dim,))
        self._Ku = self.fc.real_field("to_Ku", (self.dim,))

        self.preconditioner_kind = preconditioner
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        self._prec = None
        self._nb_pixels = tuple(self.engine.nb_subdomain_grid_pts)

    # -- geometry helpers ---------------------------------------------------
    @property
    def nb_pixels(self):
        """Local (owned) pixel grid shape -- the shape of an element-wise
        density array on this rank."""
        return self._nb_pixels

    def scalar_field(self, name):
        return self.fc.real_field(name)

    def vector_field(self, name):
        return self.fc.real_field(name, (self.dim,))

    # -- material update ----------------------------------------------------
    def set_density(self, rho):
        """Interpolate rho -> (lambda, mu), fill ghosts, and (re)build/refresh
        the preconditioner. ``rho`` has shape :attr:`nb_pixels`."""
        lam, mu = self.material.lame(np.asarray(rho))
        self.lam.p[...] = lam
        self.mu.p[...] = mu
        self.engine.communicate_ghosts(self.lam)
        self.engine.communicate_ghosts(self.mu)
        # Stiffness scale (max |λ|, |2μ|), reduced across ranks, used to detect a
        # negligible (round-off-only) right-hand side.
        local = max(float(np.abs(lam).max()), float(2.0 * np.abs(mu).max()))
        self._mat_scale = self.comm.max(local)
        self._update_preconditioner()

    def _reference_lame(self):
        # Global (MPI-reduced) means -> a rank-consistent reference material.
        n = self.comm.sum(int(self.lam.p.size))
        lam_ref = self.comm.sum(float(np.asarray(self.lam.p).sum())) / n
        mu_ref = self.comm.sum(float(np.asarray(self.mu.p).sum())) / n
        return lam_ref, mu_ref

    def _update_preconditioner(self):
        if self.preconditioner_kind is None:
            self._prec = None
            return
        if self._prec is None:
            lam_ref, mu_ref = self._reference_lame()
            if self.preconditioner_kind == "green-jacobi":
                self._prec = make_green_jacobi_preconditioner(
                    self.engine, self.op, self.lam, self.mu, self.dim,
                    reference_lambda=lam_ref, reference_mu=mu_ref,
                )
            elif self.preconditioner_kind == "green":

                def apply_ref(u, f):
                    self.engine.communicate_ghosts(u)
                    self.op.apply_uniform(u, lam_ref, mu_ref, f)

                self._prec = make_reference_stiffness_preconditioner(
                    self.engine, apply_ref, self.dim
                )
            else:
                raise ValueError(
                    f"unknown preconditioner '{self.preconditioner_kind}'"
                )
        elif self.preconditioner_kind == "green-jacobi":
            # Reuse the (reference) Green part; refresh only the Jacobi diagonal
            # from the updated material.
            self._prec.refresh()

    # -- solves -------------------------------------------------------------
    def _hessp(self, u, Au):
        self.engine.communicate_ghosts(u)
        self.op.apply(u, self.lam, self.mu, Au)

    def solve_rhs(self, b, x, rtol=None, maxiter=None, rhs_scale=None):
        """Solve ``K x = b`` in place; returns ``x``.

        A negligible right-hand side (e.g. a spatially uniform material, whose
        macro response needs no fluctuation, giving a round-off-level rhs) has
        the exact solution ``x = 0``; return it directly, since CG would divide
        by zero on the first step. ``rhs_scale`` sets the (absolute) force scale
        below which the rhs counts as zero; when omitted the material scale is
        used."""
        x.set_zero()
        b_norm = np.sqrt(self.comm.sum(
            float(np.dot(np.asarray(b.p).ravel(), np.asarray(b.p).ravel()))
        ))
        scale = rhs_scale if rhs_scale is not None else getattr(
            self, "_mat_scale", 1.0)
        if b_norm <= 1e-9 * scale:
            return x
        conjugate_gradients(
            self.comm, self.fc, b, x,
            hessp=self._hessp, prec=self._prec,
            rtol=self.cg_tol if rtol is None else rtol,
            maxiter=self.cg_maxiter if maxiter is None else maxiter,
        )
        return x

    def solve_macro(self, E_macro, x, rtol=None, maxiter=None):
        """Solve the periodic homogenization problem ``K u = -Bᵀ C:Ē`` for the
        fluctuation displacement ``u`` under macro strain ``Ē``."""
        E_arr = np.asarray(E_macro, dtype=float)
        E = list(E_arr.ravel())
        self.op.apply_macro_rhs(self.lam, self.mu, E, self._rhs)
        self._rhs.s[...] *= -1.0
        scale = getattr(self, "_mat_scale", 1.0) * max(
            float(np.abs(E_arr).max()), 1e-300)
        return self.solve_rhs(
            self._rhs, x, rtol=rtol, maxiter=maxiter, rhs_scale=scale)

    @property
    def mat_scale(self):
        """Stiffness force scale (max |λ|, |2μ|) of the current material."""
        return getattr(self, "_mat_scale", 1.0)

    def macro_rhs_tensor(self, tensor, out, scale=1.0):
        """Assemble ``out = scale * Bᵀ C : T`` for a constant symmetric tensor
        ``T`` (used to build the adjoint right-hand side)."""
        T = list(np.asarray(tensor, dtype=float).ravel())
        self.op.apply_macro_rhs(self.lam, self.mu, T, out)
        if scale != 1.0:
            out.s[...] *= scale
        return out

    def homogenized_stress(self, u, E_macro):
        """Cell-averaged stress ``⟨σ⟩ = (1/V) ∫ C:(Ē + ∇u) dV`` as a
        ``dim x dim`` array (MPI-reduced)."""
        E = list(np.asarray(E_macro, dtype=float).ravel())
        local = self.op.average_stress(u, self.lam, self.mu, E)
        glob = np.array([self.comm.sum(float(v)) for v in local])
        return glob.reshape(self.dim, self.dim) / self.domain_volume
