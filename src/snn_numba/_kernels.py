"""Numba-accelerated kernels for SNN fixed-radius search.

The expensive part of an SNN query is the *candidate scan*: for every query we
have already pruned the data set to a contiguous "slab" of points whose
projection onto the first principal component lies within ``r`` of the query's
projection.  We then need the exact squared Euclidean distance to each candidate
and keep the ones within ``r``.

These kernels fuse the dot-product, the ``||x||^2 + ||q||^2 - 2 x.q`` distance
expansion and the radius test into a single pass with no temporary allocations,
and parallelise across queries with ``prange``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def radius_fill(
    data,        # (n, d) sorted, mean-centered data (C-contiguous)
    norms,       # (n,)   squared norms of each sorted row
    sort_id,     # (n,)   original index of each sorted row
    queries,     # (m, d) mean-centered queries (C-contiguous)
    qnorms,      # (m,)   squared norms of each query
    left,        # (m,)   slab start (inclusive) per query
    right,       # (m,)   slab end   (exclusive) per query
    r2,          # (m,)   squared radius per query
    offsets,     # (m,)   write offset into the flat output buffers per query
    out_idx,     # (total,) flat output buffer for matched original indices
    out_dist,    # (total,) flat output buffer for matched squared distances
    counts,      # (m,)   number of matches written per query (output)
    want_dist,   # bool   whether to fill out_dist
):
    """Scan each query's candidate slab and write matches to flat buffers.

    Each query owns the segment ``out_idx[offsets[j] : offsets[j] + counts[j]]``
    after the call.  ``offsets`` is sized so segments never overlap (the slab
    width is an upper bound on the number of matches).
    """
    m = queries.shape[0]
    d = queries.shape[1]
    for j in prange(m):
        base = offsets[j]
        cnt = 0
        qn = qnorms[j]
        thr = r2[j]
        lo = left[j]
        hi = right[j]
        for i in range(lo, hi):
            dot = 0.0
            for k in range(d):
                dot += data[i, k] * queries[j, k]
            dist2 = norms[i] + qn - 2.0 * dot
            if dist2 <= thr:
                pos = base + cnt
                out_idx[pos] = sort_id[i]
                if want_dist:
                    out_dist[pos] = dist2 if dist2 > 0.0 else 0.0
                cnt += 1
        counts[j] = cnt


@njit(cache=True, fastmath=True, parallel=True)
def radius_count(
    data, norms, queries, qnorms, left, right, r2, counts
):
    """Like :func:`radius_fill` but only counts matches (no index output).

    Used by :meth:`SNN.query_radius` when ``count_only=True`` so we never
    materialise the index buffers.
    """
    m = queries.shape[0]
    d = queries.shape[1]
    for j in prange(m):
        cnt = 0
        qn = qnorms[j]
        thr = r2[j]
        lo = left[j]
        hi = right[j]
        for i in range(lo, hi):
            dot = 0.0
            for k in range(d):
                dot += data[i, k] * queries[j, k]
            dist2 = norms[i] + qn - 2.0 * dot
            if dist2 <= thr:
                cnt += 1
        counts[j] = cnt
