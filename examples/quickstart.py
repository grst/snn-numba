"""Quickstart: snn_numba vs sklearn KDTree on the README example."""

import time

import numpy as np
from sklearn.neighbors import KDTree

from snn_numba import SNN

rng = np.random.default_rng(0)
X = rng.random((100_000, 100))
radius = 3.5

# --- build the SNN index ---
t = time.perf_counter()
snn = SNN(X)
print(f"SNN index time:    {time.perf_counter() - t:.3f}s")

# warm up Numba JIT (compiled once), then query
snn.query_radius(X[:1], radius)

t = time.perf_counter()
ind, dist = snn.query_radius(X[0], radius, return_distance=True)
print(f"SNN query time:    {time.perf_counter() - t:.4f}s")
order = np.argsort(dist)
print("number of neighbors:", len(ind))
print("closest five:", ", ".join(str(i) for i in ind[order][:5]))

# --- compare with KDTree ---
t = time.perf_counter()
tree = KDTree(X)
print(f"\nKDTree index time: {time.perf_counter() - t:.3f}s")
t = time.perf_counter()
ind2 = tree.query_radius(X[0].reshape(1, -1), radius)[0]
print(f"KDTree query time: {time.perf_counter() - t:.4f}s")
print("number of neighbors:", len(ind2))

assert np.array_equal(np.sort(ind), np.sort(ind2)), "SNN and KDTree disagree!"
print("\nSNN and KDTree return identical neighbor sets ✓")
