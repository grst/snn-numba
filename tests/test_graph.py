"""Tests for the squidpy-independent radius-graph core (snn_radius_graph)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.neighbors import radius_neighbors_graph

from snn_numba import snn_radius_graph


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_matches_brute_force(d, dtype):
    rng = np.random.default_rng(0)
    coords = rng.random((600, d))
    r = 0.13

    adj, dst = snn_radius_graph(coords, r, set_diag=False, dtype=dtype)

    D = cdist(coords, coords)
    ref = (D <= r) & ~np.eye(len(coords), dtype=bool)
    A = adj.toarray() > 0

    assert A.shape == ref.shape
    assert (A == ref).all()
    # distances correct on the edges
    assert np.allclose(dst.toarray()[ref], D[ref], atol=1e-4 if dtype == np.float32 else 1e-9)


def test_symmetric():
    rng = np.random.default_rng(1)
    coords = rng.random((400, 2))
    adj, dst = snn_radius_graph(coords, 0.15)
    A = adj.toarray()
    assert (A == A.T).all(), "radius graph must be symmetric"
    assert adj.nnz == dst.nnz, "adj and dst must share the same sparsity pattern"


def test_set_diag():
    rng = np.random.default_rng(2)
    coords = rng.random((300, 2))
    r = 0.15

    adj0, dst0 = snn_radius_graph(coords, r, set_diag=False)
    assert adj0.diagonal().sum() == 0, "no self-loops when set_diag=False"

    adj1, dst1 = snn_radius_graph(coords, r, set_diag=True)
    assert (adj1.diagonal() == 1).all(), "self-loops present when set_diag=True"
    assert (dst1.diagonal() == 0).all(), "self distance pinned to exactly 0"
    # the only difference between the two should be the diagonal
    assert adj1.nnz == adj0.nnz + len(coords)


def test_matches_sklearn_radius_graph():
    rng = np.random.default_rng(3)
    coords = rng.random((500, 2))
    r = 0.1
    adj, _ = snn_radius_graph(coords, r, set_diag=False)
    sk = radius_neighbors_graph(coords, r, mode="connectivity", include_self=False)
    assert ((adj.toarray() > 0) == (sk.toarray() > 0)).all()


def test_empty_graph_small_radius():
    rng = np.random.default_rng(4)
    coords = rng.random((100, 2)) * 1000  # spread out
    adj, dst = snn_radius_graph(coords, 1e-6, set_diag=False)
    assert adj.nnz == 0
    assert dst.nnz == 0
    assert adj.shape == (100, 100)


def test_invalid_inputs():
    coords = np.random.default_rng(0).random((10, 2))
    with pytest.raises(ValueError):
        snn_radius_graph(coords, 0.0)
    with pytest.raises(ValueError):
        snn_radius_graph(np.zeros(10), 0.5)  # not 2D
