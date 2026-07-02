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

from itertools import combinations_with_replacement, product
from math import factorial

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


def _gauss_points_unit_interval():
    """3-point Gauss rule mapped to [0, 1] (exact to polynomial degree 5)."""
    g = np.sqrt(3.0 / 5.0)
    pts = [(0.5 * (1.0 - g), 5.0 / 18.0), (0.5, 8.0 / 18.0),
           (0.5 * (1.0 + g), 5.0 / 18.0)]
    return pts


class ConsistentDoubleWell:
    """Exact Galerkin integral of ``W(rho) = rho^2 (1-rho)^2`` of the nodal FE
    interpolant, with its exact nodal gradient.

    P1 (simplices): closed form. On a ``d``-simplex ``T`` with corner values
    ``a``, ``\\int_T rho^k = d! |T| k!/(k+d)! h_k(a)`` with the complete
    homogeneous symmetric polynomial ``h_k``, so
    ``\\int_T W = |T| [C2 h2 - 2 C3 h3 + C4 h4]``, ``Ck = d! k!/(k+d)!``.

    Q1 (multilinear): 3-point tensor Gauss quadrature, exact for the
    degree-(4 per axis) integrand.
    """

    def __init__(self, nodal_map: NodalElementMap):
        self.m = nodal_map
        dim = self.m.dim
        if self.m.element_name == "p1":
            m_nodes = dim + 1
            self._msets = {
                k: list(combinations_with_replacement(range(m_nodes), k))
                for k in (2, 3, 4)
            }
            self._Ck = {
                k: factorial(dim) * factorial(k) / factorial(k + dim)
                for k in (2, 3, 4)
            }
        else:  # q1
            pts1d = _gauss_points_unit_interval()
            self._gauss = []
            for combo in product(pts1d, repeat=dim):
                xi = [c[0] for c in combo]
                w = float(np.prod([c[1] for c in combo]))
                # Multilinear shape function of corner c at xi.
                N = [
                    float(np.prod([
                        xi[d] if _node_offset(c, d) else 1.0 - xi[d]
                        for d in range(dim)
                    ]))
                    for c in range(self.m.nb_nodes)
                ]
                self._gauss.append((w, N))

    # -- P1: closed form via h_k ---------------------------------------------
    def _p1_value_and_corner_grad(self, views):
        m = self.m
        f = None
        grads = [None] * m.nb_nodes
        for nodes, frac in _P1_SIMPLICES[m.dim]:
            V = frac * m.vol_pixel
            a = [views[i] for i in nodes]
            # h_k and dh_k/da_j, accumulated into the W combination
            # W = C2 h2 - 2 C3 h3 + C4 h4 (times V).
            for k, coeff in ((2, self._Ck[2]), (3, -2.0 * self._Ck[3]),
                             (4, self._Ck[4])):
                for t in self._msets[k]:
                    prod = a[t[0]]
                    for i in t[1:]:
                        prod = prod * a[i]
                    f = V * coeff * prod if f is None else f + V * coeff * prod
                    # gradient: for each distinct entry j of t with
                    # multiplicity mult, d(prod)/da_j = mult * prod(t \ one j)
                    for j in set(t):
                        rem = None
                        removed = False
                        for i in t:
                            if i == j and not removed:
                                removed = True
                                continue
                            rem = a[i] if rem is None else rem * a[i]
                        if rem is None:
                            rem = 1.0
                        term = V * coeff * t.count(j) * rem
                        g = grads[nodes[j]]
                        grads[nodes[j]] = term if g is None else g + term
        return f, grads

    # -- Q1: exact Gauss quadrature --------------------------------------------
    def _q1_value_and_corner_grad(self, views):
        m = self.m
        f = None
        grads = [None] * m.nb_nodes
        for w, N in self._gauss:
            rho_g = N[0] * views[0]
            for c in range(1, m.nb_nodes):
                rho_g = rho_g + N[c] * views[c]
            Wg = rho_g ** 2 * (1.0 - rho_g) ** 2
            dWg = 2.0 * rho_g * (1.0 - rho_g) * (1.0 - 2.0 * rho_g)
            fac = w * m.vol_pixel
            f = fac * Wg if f is None else f + fac * Wg
            for c in range(m.nb_nodes):
                term = fac * N[c] * dWg
                grads[c] = term if grads[c] is None else grads[c] + term
        return f, grads

    def value_and_gradient(self, rho):
        """Return ``(\\int W dx, d/drho)`` for a nodal density array; the value
        is MPI-reduced, the gradient is the local nodal slice."""
        m = self.m
        views = m.corner_values(rho)
        if m.element_name == "p1":
            f_field, grads = self._p1_value_and_corner_grad(views)
        else:
            f_field, grads = self._q1_value_and_corner_grad(views)
        f = m.h.comm.sum(float(np.sum(m.h.to_host(f_field))))
        grad = m.scatter(grads)
        return f, grad
