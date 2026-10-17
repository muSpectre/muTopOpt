#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""
Nodal design fields: the density is a *nodal* finite-element field (one degree
of freedom per grid node) rather than an element-wise (per-pixel) constant.

Two ingredients live here:

* :class:`NodalElementMap` -- the Galerkin-consistent coupling between the
  nodal density and the per-element material the fused stiffness operator
  consumes. The element density is the exact element average of the FE
  interpolant, ``rho_e = (1/|e|) \\int_e rho(x) dx = sum_c w_c rho_{n(e,c)}``,
  with weights ``w_c`` computed from the element decomposition (uniform corner
  weights for Q1; sub-simplex volume weights for P1). Because every nodal
  degree of freedom influences its ``2^dim`` adjacent elements, this coupling
  acts as an implicit sensitivity filter -- the property that lets the
  optimizer merge or dissolve features instead of locking in the initial
  topology. The adjoint (``scatter``) is the exact transpose of the gather and
  is MPI-correct: contributions to ghost nodes are folded back onto their
  owners by muGrid's ghost reduction.

* :class:`ConsistentDoubleWell` -- the double-well energy
  ``\\int W(rho) dx``, ``W = rho^2 (1-rho)^2``, of the nodal interpolant,
  integrated *exactly* (fully consistent Galerkin, not lumped): in closed form
  on P1 simplices via complete homogeneous symmetric polynomials
  (``\\int_T rho^k = d! |T| k!/(k+d)! h_k(a)`` for corner values ``a``), and by
  3-point tensor Gauss quadrature (exact for the degree-4-per-axis integrand)
  on Q1 elements.
"""

import muGrid
import numpy as np

#: Sub-simplex decompositions used by the P1 elements (matching
#: muGrid/operators/fem_element.hh): (corner-node ids, volume fraction).
#: Node ids are binary corner indices, x fastest: node = x + 2 y (+ 4 z).
_P1_SIMPLICES = {
    2: [((0, 1, 2), 0.5), ((1, 2, 3), 0.5)],
    3: [((1, 2, 4, 7), 1.0 / 3.0), ((0, 1, 2, 4), 1.0 / 6.0),
        ((1, 2, 3, 7), 1.0 / 6.0), ((1, 4, 5, 7), 1.0 / 6.0),
        ((2, 4, 6, 7), 1.0 / 6.0)],
}


def _node_offset(n, d):
    """Binary corner offset (0 or 1) of node ``n`` along axis ``d``
    (x fastest), matching muGrid's fem_node_offset."""
    return (n >> d) & 1


def element_average_weights(element_name, dim):
    """Exact element-average weights ``w_c = (1/|e|) \\int_e N_c dx`` of the FE
    interpolant, per binary corner node."""
    nb_nodes = 2 ** dim
    if element_name == "q1":
        # Multilinear shape functions: every corner integrates to |e|/2^dim.
        return np.full(nb_nodes, 1.0 / nb_nodes)
    if element_name == "p1":
        # Sum |T|/(d+1) over the sub-simplices containing each node.
        w = np.zeros(nb_nodes)
        for nodes, frac in _P1_SIMPLICES[dim]:
            for i in nodes:
                w[i] += frac / (dim + 1)
        return w
    raise ValueError(f"unknown element '{element_name}'")


class NodalElementMap:
    """Gather nodal densities to element averages and scatter element
    sensitivities back to nodes (the exact adjoint), across MPI ranks.

    On the periodic grid there is one node per pixel (the pixel's lower-left
    corner), so nodal arrays have the same shape as element arrays
    (:attr:`Homogenization.nb_pixels`). Element ``i`` touches nodes
    ``i + offset(c)`` for the ``2^dim`` binary corner offsets; neighbor values
    across rank (and periodic) boundaries travel through the fields' ghost
    layers.
    """

    def __init__(self, homogenization):
        self.h = homogenization
        self.dim = self.h.dim
        self.nb_nodes = 2 ** self.dim
        self.element_name = self.h.element_name
        self.vol_pixel = float(np.prod(self.h.grid_spacing))
        self.avg_weights = element_average_weights(self.element_name, self.dim)

        self._nodal = self.h.scalar_field("to_nodal_map_rho")
        self._acc = self.h.scalar_field("to_nodal_map_acc")

    # -- ghosted corner views -------------------------------------------------
    def _corner_slices(self):
        n = self.h.nb_pixels
        slices = []
        for c in range(self.nb_nodes):
            slices.append(tuple(
                slice(1 + _node_offset(c, d), 1 + _node_offset(c, d) + n[d])
                for d in range(self.dim)
            ))
        return slices

    def corner_values(self, rho):
        """Load a nodal density array, fill the ghost layer, and return the
        ``2^dim`` per-corner arrays (each shaped like the element grid, in the
        fields' array module)."""
        self._nodal.p[...] = self.h.to_device(np.asarray(rho, dtype=float))
        self.h.engine.communicate_ghosts(self._nodal)
        pg = self._nodal.pg
        return [pg[sl] for sl in self._corner_slices()]

    # -- gather / scatter -----------------------------------------------------
    def gather_mean(self, rho):
        """Element averages ``rho_e = sum_c w_c rho_{n(e,c)}`` as a host NumPy
        array shaped like the element grid."""
        views = self.corner_values(rho)
        acc = self.avg_weights[0] * views[0]
        for c in range(1, self.nb_nodes):
            acc = acc + self.avg_weights[c] * views[c]
        return self.h.to_host(acc)

    def scatter(self, per_corner):
        """Adjoint of corner gathering: node ``i + offset(c)`` accumulates
        ``per_corner[c][i]``; ghost-node contributions are reduced back onto
        their owning rank. Returns a host NumPy nodal array."""
        pg = self._acc.pg
        pg[...] = 0.0
        for sl, contrib in zip(self._corner_slices(), per_corner):
            pg[sl] += contrib
        self.h.engine.reduce_ghosts(self._acc)
        return self.h.to_host(self._acc.p).copy()

    def scatter_mean(self, s_e):
        """Adjoint of :meth:`gather_mean` for an element array ``s_e``:
        ``grad_n = sum_e w_c s_e`` over the adjacent elements."""
        s = self.h.to_device(np.asarray(s_e, dtype=float))
        return self.scatter([w * s for w in self.avg_weights])


class ConsistentDoubleWell:
    """Exact Galerkin integral of ``W(rho) = rho^2 (1-rho)^2`` of the nodal FE
    interpolant, with its exact nodal gradient.

    ``W`` is quartic, so its cell integral is a fixed combination of the
    moments ``M_k = int_e rho(x)^k dx`` of the interpolant:
    ``rho^2 (1-rho)^2 = rho^2 - 2 rho^3 + rho^4``, hence
    ``int_e W = M2 - 2 M3 + M4``, and the nodal gradient is the same
    combination of the moment gradients.

    muGrid's fused :class:`NodalMomentOperator` computes both in a single
    pass, exactly for either element (3-point-per-axis tensor Gauss on Q1, the
    same rule per sub-simplex on P1). Evaluating the quadrature
    array-at-a-time instead -- the previous implementation here -- materialises
    the interpolant at every quadrature point of every cell at once, which in
    3D is a 27-fold copy of the grid plus a temporary per polynomial term, and
    made this term the dominant memory consumer of a whole optimisation.
    """

    #: Number of moments the operator returns, and the coefficients of W on
    #: them (M2, M3, M4).
    NB_MOMENTS = 3
    W_COEFFS = np.array([1.0, -2.0, 1.0])

    def __init__(self, nodal_map: NodalElementMap):
        self.m = nodal_map
        self.h = nodal_map.h
        element = (muGrid.FEMElement.p1 if self.m.element_name == "p1"
                   else muGrid.FEMElement.q1)
        op_cls = (muGrid.NodalMomentOperator2D if self.m.dim == 2
                  else muGrid.NodalMomentOperator3D)
        self.op = op_cls(list(self.h.grid_spacing), element)
        # Scratch: the nodal input (ghosts filled per call) and the operator's
        # two multi-component outputs. Three scalar-field-sized buffers in
        # total, replacing the transient 27-fold grid copies.
        self._rho = self.h.scalar_field("to_dwell_rho")
        self._moments = self.h.fc.real_field(
            "to_dwell_moments", (self.NB_MOMENTS,), dtype=self.h.dtype)
        self._grads = self.h.fc.real_field(
            "to_dwell_moment_grads", (self.NB_MOMENTS,), dtype=self.h.dtype)

    def value_and_gradient(self, rho):
        """Return ``(int W dx, d/drho)`` for a nodal density array; the value
        is MPI-reduced, the gradient is the local nodal slice.

        The kernel gives each thread its own node, so it writes the nodal
        gradient directly -- no scatter and no ghost reduction here.
        """
        h = self.h
        self._rho.p[...] = h.to_device(np.asarray(rho, dtype=h.dtype))
        h.engine.communicate_ghosts(self._rho)
        self.op.compute(self._rho, self._moments, self._grads)
        c = self.W_COEFFS
        m = h.to_host(self._moments.p).reshape((self.NB_MOMENTS, -1))
        g = h.to_host(self._grads.p).reshape((self.NB_MOMENTS, -1))
        f = h.comm.sum(float(c @ m.sum(axis=1)))
        return f, (c @ g).reshape(np.shape(rho))
