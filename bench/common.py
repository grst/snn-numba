"""Shared helpers for the SNN benchmarks: data, timing, radius tuning."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# make the locally-compiled original C++ module importable
_CPP_DIR = os.path.join(os.path.dirname(__file__), "_cpp")
if _CPP_DIR not in sys.path:
    sys.path.insert(0, _CPP_DIR)


def load_cpp():
    """Import the original nla-group/snn C++ module, or return None if missing."""
    try:
        import snnomp  # noqa: F401

        return snnomp
    except Exception as exc:  # pragma: no cover
        print(f"[warn] C++ snnomp not available: {exc}", file=sys.stderr)
        return None


def make_data(n, d, seed=0, n_clusters=None):
    """Generate (X, queries) test data.

    Uniform data is hard for *every* method (no structure to exploit); clustered
    data is more representative of real workloads.  ``n_clusters`` enables
    Gaussian clusters; otherwise uniform [0, 1]^d.
    """
    rng = np.random.default_rng(seed)
    if n_clusters:
        centers = rng.random((n_clusters, d))
        labels = rng.integers(0, n_clusters, n)
        X = centers[labels] + rng.normal(0, 0.08, (n, d))
    else:
        X = rng.random((n, d))
    return np.ascontiguousarray(X, dtype=np.float64)


def make_queries(X, m, seed=1):
    """Pick ``m`` query points near the data manifold (perturbed samples)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    idx = rng.integers(0, n, m)
    return np.ascontiguousarray(X[idx] + rng.normal(0, 0.02, (m, d)), dtype=np.float64)


def tune_radius(snn_model, queries, target_neighbors, lo=1e-4, hi=None, iters=30):
    """Bisect a radius so queries return ~``target_neighbors`` neighbors on average.

    Uses SNN's cheap ``count_only`` path so tuning is fast and method-agnostic.
    """
    if hi is None:
        # a generous upper bound: diameter-ish
        hi = float(np.linalg.norm(queries.max(0) - queries.min(0))) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        avg = snn_model.query_radius(queries, mid, count_only=True).mean()
        if avg < target_neighbors:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def timeit(fn, repeat=5, number=1):
    """Return the best wall-clock time of ``fn`` over ``repeat`` runs (seconds)."""
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            out = fn()
        dt = (time.perf_counter() - t0) / number
        best = min(best, dt)
    return best, out
