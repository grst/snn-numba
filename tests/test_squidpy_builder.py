"""Integration test for the squidpy SNNRadiusBuilder extension.

Skipped automatically if squidpy (with the builder API) is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

squidpy = pytest.importorskip("squidpy")
anndata = pytest.importorskip("anndata")

# the builder API itself is recent; skip cleanly if this squidpy lacks it
if not hasattr(squidpy.gr, "spatial_neighbors_from_builder"):
    pytest.skip(
        "installed squidpy lacks spatial_neighbors_from_builder",
        allow_module_level=True,
    )

from snn_numba.squidpy import SNNRadiusBuilder, spatial_neighbors  # noqa: E402


def _toy_adata(n=300, d=2, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.random((n, d)) * 100.0
    X = rng.random((n, 5))
    adata = anndata.AnnData(X)
    adata.obsm["spatial"] = coords
    return adata, coords


def test_builder_matches_brute_force_radius_graph():
    adata, coords = _toy_adata()
    r = 12.0
    squidpy.gr.spatial_neighbors_from_builder(adata, SNNRadiusBuilder(radius=r))

    conn = adata.obsp["spatial_connectivities"].toarray() > 0
    D = cdist(coords, coords)
    ref = (D <= r) & ~np.eye(len(coords), dtype=bool)
    assert (conn == ref).all()

    dist = adata.obsp["spatial_distances"].toarray()
    assert np.allclose(dist[ref], D[ref], atol=1e-9)


def test_convenience_wrapper_and_uns():
    adata, coords = _toy_adata()
    r = 10.0
    spatial_neighbors(adata, radius=r)
    assert "spatial_connectivities" in adata.obsp
    assert "spatial_distances" in adata.obsp
    # uns params recorded
    uns = adata.uns.get("spatial", {})
    assert any(
        isinstance(v, dict) and v.get("radius") == r for v in _walk(uns)
    ) or uns.get("radius") == r


def _walk(obj):
    """Yield obj and nested dict values (uns layout varies across versions)."""
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
