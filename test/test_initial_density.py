#
# Copyright 2026 Lars Pastewka
#
# MIT License (see LICENSE)
#
"""The initial design must not depend on the domain decomposition."""

import numpy as np
import pytest

from muTopOpt.optimize import initial_density


@pytest.mark.parametrize("kind", ["uniform", "random", "filtered_random"])
@pytest.mark.parametrize("nb_ranks", [2, 3, 4])
def test_subdomains_tile_the_global_field(kind, nb_ranks):
    """Slab subdomains, as the FFT engine splits the grid, reassemble to
    exactly the field generated on the whole grid."""
    shape = (16, 12, 24)
    kwargs = dict(kind=kind, seed=5, length=0.1, volume_fraction=0.4)
    glob = initial_density(shape, **kwargs)
    nz = shape[-1] // nb_ranks
    pieces = [
        initial_density(shape, subdomain_locations=(0, 0, r * nz),
                        nb_subdomain_grid_pts=shape[:-1] + (nz,), **kwargs)
        for r in range(nb_ranks)
    ]
    np.testing.assert_array_equal(np.concatenate(pieces, axis=-1), glob)


def test_subdomain_needs_both_arguments():
    with pytest.raises(ValueError):
        initial_density((8, 8), subdomain_locations=(0, 0))
