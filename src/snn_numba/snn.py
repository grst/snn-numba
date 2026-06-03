"""Pure Python/Numba implementation of SNN fixed-radius nearest-neighbor search.

SNN ("sorting-based nearest neighbors") is a *fast and exact* fixed-radius
neighbor search algorithm:

    Chen X, Güttel S. 2024. Fast and exact fixed-radius neighbor search based on
    sorting. PeerJ Computer Science 10:e1929. https://doi.org/10.7717/peerj-cs.1929

Original C++/Python reference implementation: https://github.com/nla-group/snn

Idea
----
Projection onto a unit vector ``v`` is 1-Lipschitz::

    |v.x - v.q| <= ||x - q||

so *any* point ``x`` within radius ``r`` of a query ``q`` must have its
projection within ``r`` of ``q``'s projection.  SNN picks ``v`` = first
principal component of the (mean-centered) data, sorts all points by their
projection, and at query time uses two binary searches to restrict the exact
distance computation to the thin "slab" of points whose projection lies in
``[v.q - r, v.q + r]``.  Because the slab is a *necessary* condition and the
final filter uses exact Euclidean distance, the result is exact.

The API mirrors :class:`sklearn.neighbors.KDTree`.
"""

from __future__ import annotations

import numpy as np

from . import _kernels


def _first_principal_component(Xc, n_iter=100, tol=1e-6, seed=0):
    """Dominant right singular vector of the centered data via power iteration.

    Iterates ``v <- Xc.T (Xc v)`` (BLAS GEMVs) which converges to the top
    eigenvector of the covariance ``Xc.T Xc`` (= first principal component) and
    stops early once the direction stabilises.

    Note: SNN is *exact* for **any** projection direction ``v`` (the slab is
    always a valid superset of the true neighbors).  ``v`` only controls how
    tight the pruning is, so a few iterations / a rough estimate are enough --
    we never need to fully converge, which matters for isotropic data where the
    top eigenvalues are nearly equal and power iteration converges slowly.
    """
    d = Xc.shape[1]
    if d == 1:
        return np.ones(1, dtype=Xc.dtype)

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(Xc.dtype)
    v /= np.linalg.norm(v)

    for _ in range(n_iter):
        w = Xc.T @ (Xc @ v)
        nrm = np.linalg.norm(w)
        if nrm < 1e-30:
            break  # degenerate (e.g. all points identical); direction irrelevant
        w /= nrm
        # converged if direction no longer changes (sign-insensitive)
        if 1.0 - abs(float(w @ v)) < tol:
            v = w
            break
        v = w
    return v.astype(Xc.dtype)


class SNN:
    """Sorting-based fast, exact fixed-radius neighbor search.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to index.
    dtype : {numpy.float32, numpy.float64}, default numpy.float64
        Working precision.  ``float32`` roughly halves memory traffic and is
        usually noticeably faster, at the cost of float rounding near the radius
        boundary.
    n_iter : int, default 100
        Maximum power-iteration steps for the first principal component.
    pc_sample : int, default 20000
        If ``n_samples`` exceeds this, the principal component is estimated from
        a random subsample of this many rows (it is only used for pruning, so an
        estimate keeps the build fast and ``n``-independent without affecting
        correctness).  Set to 0 to always use all rows.

    Attributes
    ----------
    mean_ : ndarray (n_features,)
        Column means subtracted from the data.
    components_ : ndarray (n_features,)
        First principal component (unit vector) used for projection.

    Notes
    -----
    The interface mirrors :class:`sklearn.neighbors.KDTree`: construct on the
    data, then call :meth:`query_radius`.
    """

    def __init__(self, X, dtype=np.float64, n_iter=100, pc_sample=20000):
        X = np.ascontiguousarray(X, dtype=dtype)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        self.dtype = np.dtype(dtype)
        self.n_samples_, self.n_features_ = X.shape

        # 1. center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 2. first principal component (estimated on a subsample for large n;
        #    only affects pruning tightness, never correctness)
        if pc_sample and self.n_samples_ > pc_sample:
            sample_idx = np.random.default_rng(0).choice(
                self.n_samples_, pc_sample, replace=False
            )
            self.components_ = _first_principal_component(Xc[sample_idx], n_iter=n_iter)
        else:
            self.components_ = _first_principal_component(Xc, n_iter=n_iter)

        # 3. project and sort
        proj = Xc @ self.components_
        order = np.argsort(proj, kind="stable")
        self._sort_id = order.astype(np.int64)
        self._proj_sorted = np.ascontiguousarray(proj[order])
        self._data_sorted = np.ascontiguousarray(Xc[order])
        # squared norm of every (centered) row, in sorted order
        self._norms = np.einsum(
            "ij,ij->i", self._data_sorted, self._data_sorted
        ).astype(self.dtype)

    # ------------------------------------------------------------------ #
    # query
    # ------------------------------------------------------------------ #
    def query_radius(
        self,
        X,
        r,
        return_distance=False,
        count_only=False,
        sort_results=False,
    ):
        """Find all neighbors within distance ``r`` of each point in ``X``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features) or (n_features,)
            Query point(s).  A single 1D point is accepted and treated as one
            query.
        r : float or array-like of shape (n_queries,)
            Search radius (or per-query radii).
        return_distance : bool, default False
            If True, also return the Euclidean distance to each neighbor.
        count_only : bool, default False
            If True, return only the number of neighbors per query (an int
            array).  Cannot be combined with ``return_distance`` /
            ``sort_results``.
        sort_results : bool, default False
            If True, sort each query's neighbors by increasing distance.
            Requires ``return_distance=True`` (as in scikit-learn).

        Returns
        -------
        ind : ndarray of object, shape (n_queries,)
            ``ind[i]`` is an int array of neighbor indices for query ``i``.
        dist : ndarray of object, shape (n_queries,)
            Returned only if ``return_distance=True``; ``dist[i]`` are the
            matching Euclidean distances.
        count : ndarray of int, shape (n_queries,)
            Returned instead of ``ind`` if ``count_only=True``.
        """
        if count_only and (return_distance or sort_results):
            raise ValueError(
                "count_only cannot be combined with return_distance/sort_results"
            )
        if sort_results and not return_distance:
            raise ValueError("sort_results=True requires return_distance=True")

        X = np.asarray(X, dtype=self.dtype)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(
                f"X must have {self.n_features_} features, got shape {X.shape}"
            )
        m = X.shape[0]

        r = np.asarray(r, dtype=self.dtype)
        if r.ndim == 0:
            r = np.full(m, float(r), dtype=self.dtype)
        elif r.shape != (m,):
            raise ValueError("r must be a scalar or have one value per query")
        r2 = (r * r).astype(self.dtype)

        # center & project queries (BLAS)
        Xc_q = np.ascontiguousarray(X - self.mean_)
        proj_q = Xc_q @ self.components_
        qnorms = np.einsum("ij,ij->i", Xc_q, Xc_q).astype(self.dtype)

        # slab boundaries via binary search on the sorted projections.
        # 'left' for the lower bound and 'right' for the upper bound makes the
        # slab inclusive of points whose projection equals q +/- r.
        left = np.searchsorted(self._proj_sorted, proj_q - r, side="left")
        right = np.searchsorted(self._proj_sorted, proj_q + r, side="right")
        left = left.astype(np.int64)
        right = right.astype(np.int64)

        if count_only:
            counts = np.empty(m, dtype=np.int64)
            _kernels.radius_count(
                self._data_sorted, self._norms, Xc_q, qnorms,
                left, right, r2, counts,
            )
            return counts[0] if single else counts

        slab_sizes = right - left
        total = int(slab_sizes.sum())
        offsets = np.zeros(m, dtype=np.int64)
        if m > 1:
            np.cumsum(slab_sizes[:-1], out=offsets[1:])

        out_idx = np.empty(total, dtype=np.int64)
        out_dist = np.empty(total, dtype=self.dtype) if return_distance else \
            np.empty(0, dtype=self.dtype)
        counts = np.empty(m, dtype=np.int64)

        _kernels.radius_fill(
            self._data_sorted, self._norms, self._sort_id, Xc_q, qnorms,
            left, right, r2, offsets, out_idx, out_dist, counts,
            return_distance,
        )

        ind = np.empty(m, dtype=object)
        dist = np.empty(m, dtype=object) if return_distance else None
        for j in range(m):
            beg = offsets[j]
            end = beg + counts[j]
            idx_j = out_idx[beg:end].copy()
            if return_distance:
                dist_j = np.sqrt(out_dist[beg:end])
                if sort_results:
                    order = np.argsort(dist_j, kind="stable")
                    idx_j = idx_j[order]
                    dist_j = dist_j[order]
                dist[j] = dist_j
            ind[j] = idx_j

        if single:
            return (ind[0], dist[0]) if return_distance else ind[0]
        return (ind, dist) if return_distance else ind

    def __repr__(self):
        return (
            f"SNN(n_samples={self.n_samples_}, n_features={self.n_features_}, "
            f"dtype={self.dtype})"
        )
