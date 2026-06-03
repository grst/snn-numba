"""Correctness tests for snn_numba.SNN, using sklearn KDTree as ground truth.

SNN is an *exact* fixed-radius method, so for float64 every query must return
exactly the same neighbor set as KDTree.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KDTree

from snn_numba import SNN


def _datasets():
    rng = np.random.default_rng(0)
    # (name, X, queries)
    yield "uniform-2d", rng.random((3000, 2)), rng.random((50, 2))
    yield "uniform-10d", rng.random((4000, 10)), rng.random((50, 10))
    yield "uniform-100d", rng.random((4000, 100)), rng.random((40, 100))
    # clustered
    centers = rng.random((8, 20))
    X = centers[rng.integers(0, 8, 5000)] + rng.normal(0, 0.1, (5000, 20))
    Q = X[rng.integers(0, 5000, 50)] + rng.normal(0, 0.05, (50, 20))
    yield "clustered-20d", X, Q


def _radius_for(X, Q, target=30):
    """Pick a radius via KDTree giving ~target neighbors on average."""
    kd = KDTree(X)
    lo, hi = 0.0, float(np.linalg.norm(X.max(0) - X.min(0))) + 1.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        avg = kd.query_radius(Q, mid, count_only=True).mean()
        lo, hi = (mid, hi) if avg < target else (lo, mid)
    return 0.5 * (lo + hi)


@pytest.mark.parametrize("name,X,Q", list(_datasets()), ids=lambda v: v if isinstance(v, str) else "")
def test_exact_match_float64(name, X, Q):
    r = _radius_for(X, Q)
    snn = SNN(X, dtype=np.float64)
    kd = KDTree(X)
    got = snn.query_radius(Q, r)
    exp = kd.query_radius(Q, r)
    assert len(got) == len(Q)
    for g, e in zip(got, exp):
        assert np.array_equal(np.sort(g), np.sort(e)), f"{name}: neighbor set differs"


def test_distances_match_and_sorted():
    rng = np.random.default_rng(3)
    X = rng.random((3000, 8))
    Q = rng.random((30, 8))
    r = _radius_for(X, Q, target=40)
    snn = SNN(X)
    kd = KDTree(X)
    gi, gd = snn.query_radius(Q, r, return_distance=True, sort_results=True)
    ei, ed = kd.query_radius(Q, r, return_distance=True, sort_results=True)
    for i in range(len(Q)):
        assert np.array_equal(gi[i], ei[i])
        assert np.allclose(gd[i], ed[i], atol=1e-9)
        # distances sorted ascending
        assert np.all(np.diff(gd[i]) >= -1e-12)
        # distances are correct Euclidean distances and within radius
        assert np.all(gd[i] <= r + 1e-9)
        assert np.allclose(gd[i], np.linalg.norm(X[gi[i]] - Q[i], axis=1), atol=1e-9)


def test_count_only_matches_index_count():
    rng = np.random.default_rng(4)
    X = rng.random((2000, 5))
    Q = rng.random((25, 5))
    r = _radius_for(X, Q)
    snn = SNN(X)
    counts = snn.query_radius(Q, r, count_only=True)
    inds = snn.query_radius(Q, r)
    assert np.array_equal(counts, [len(a) for a in inds])


def test_per_query_radius():
    rng = np.random.default_rng(5)
    X = rng.random((3000, 6))
    Q = rng.random((20, 6))
    radii = rng.uniform(0.2, 0.6, len(Q))
    snn = SNN(X)
    kd = KDTree(X)
    got = snn.query_radius(Q, radii)
    for i in range(len(Q)):
        exp = kd.query_radius(Q[i : i + 1], radii[i])[0]
        assert np.array_equal(np.sort(got[i]), np.sort(exp))


def test_single_query_returns_flat():
    rng = np.random.default_rng(6)
    X = rng.random((2000, 4))
    q = rng.random(4)
    r = 0.5
    snn = SNN(X)
    ind = snn.query_radius(q, r)
    # a 1D query yields a flat index array (like the original snnpy)
    assert ind.ndim == 1
    exp = KDTree(X).query_radius(q.reshape(1, -1), r)[0]
    assert np.array_equal(np.sort(ind), np.sort(exp))


def test_float32_matches_within_tolerance():
    rng = np.random.default_rng(7)
    X = rng.random((4000, 16))
    Q = rng.random((40, 16))
    r = _radius_for(X, Q, target=30)
    snn = SNN(X.astype(np.float32), dtype=np.float32)
    kd = KDTree(X)
    got = snn.query_radius(Q.astype(np.float32), np.float32(r))
    exp = kd.query_radius(Q, r)
    # float32 may disagree only on points exactly on the boundary; allow a tiny
    # symmetric-difference rate.
    total = sum(len(e) for e in exp)
    diff = sum(len(set(g.tolist()) ^ set(e.tolist())) for g, e in zip(got, exp))
    assert diff <= max(2, 0.001 * total), f"too many float32 boundary diffs: {diff}/{total}"


def test_input_validation():
    X = np.random.default_rng(0).random((100, 3))
    snn = SNN(X)
    with pytest.raises(ValueError):
        snn.query_radius(np.zeros((5, 4)), 0.5)  # wrong n_features
    with pytest.raises(ValueError):
        snn.query_radius(X[:5], 0.5, count_only=True, return_distance=True)
    with pytest.raises(ValueError):
        snn.query_radius(X[:5], 0.5, sort_results=True)  # needs return_distance
    with pytest.raises(ValueError):
        SNN(np.zeros(10))  # not 2D
