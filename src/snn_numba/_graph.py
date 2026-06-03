"""Build a fixed-radius spatial neighbor graph with SNN.

This is the squidpy-independent core used by
:class:`snn_numba.squidpy.SNNRadiusBuilder`.  It turns point coordinates into
the ``(adj, dst)`` pair of sparse matrices that squidpy's builder protocol
expects, but depends only on NumPy/SciPy/snn_numba so it can be used (and
tested) without squidpy installed.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .snn import SNN


def snn_radius_graph(coords, radius, set_diag=False, dtype=np.float64):
    """Construct a fixed-radius neighbor graph from spatial coordinates.

    For every point, all points within Euclidean distance ``radius`` are
    connected.  This is the *generic radius graph* squidpy builds for
    arbitrary spatial data, computed here with the exact SNN search.

    Parameters
    ----------
    coords : array-like of shape (n_obs, n_spatial_dims)
        Spatial coordinates (e.g. ``adata.obsm["spatial"]``).
    radius : float
        Connectivity radius.
    set_diag : bool, default False
        If True, keep self-loops (diagonal of ``adj`` = 1, of ``dst`` = 0).
        If False, self-loops are removed.
    dtype : {numpy.float32, numpy.float64}, default numpy.float64
        Working precision for the SNN search.

    Returns
    -------
    adj : scipy.sparse.csr_matrix, shape (n_obs, n_obs)
        Binary connectivity matrix (1.0 for connected pairs).  Symmetric, since
        a radius graph is symmetric.
    dst : scipy.sparse.csr_matrix, shape (n_obs, n_obs)
        Euclidean distances for the same edges (explicit 0.0 on the diagonal
        when ``set_diag``).
    """
    coords = np.ascontiguousarray(coords, dtype=dtype)
    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array (n_obs, n_spatial_dims)")
    n = coords.shape[0]
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")

    snn = SNN(coords, dtype=dtype)
    ind, dist = snn.query_radius(coords, radius, return_distance=True)

    # flatten the ragged per-row neighbor lists into COO triplets
    counts = np.fromiter((a.shape[0] for a in ind), count=n, dtype=np.int64)
    if counts.sum():
        rows = np.repeat(np.arange(n, dtype=np.int64), counts)
        cols = np.concatenate(ind).astype(np.int64, copy=False)
        dsts = np.concatenate(dist).astype(np.float64, copy=False)
    else:  # no edges at all (radius too small)
        rows = np.empty(0, np.int64)
        cols = np.empty(0, np.int64)
        dsts = np.empty(0, np.float64)

    if not set_diag:
        # the radius search always returns each point as its own neighbor
        # (distance 0); drop those self-loops.
        keep = rows != cols
        rows, cols, dsts = rows[keep], cols[keep], dsts[keep]

    adj = csr_matrix(
        (np.ones(rows.shape[0], dtype=np.float64), (rows, cols)),
        shape=(n, n),
    )
    dst = csr_matrix((dsts, (rows, cols)), shape=(n, n))

    if set_diag:
        # the self-distance from the search is ~0 up to float rounding; pin the
        # diagonal to exact values (adj=1, dst=0), matching squidpy's builders.
        adj.setdiag(1.0)
        dst.setdiag(0.0)

    adj.sort_indices()
    dst.sort_indices()
    return adj, dst
