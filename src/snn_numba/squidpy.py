"""Squidpy extension: build spatial neighbor graphs with SNN.

This module plugs ``snn_numba`` into squidpy's builder protocol
(https://squidpy.readthedocs.io/en/latest/extensibility.html).  It provides
:class:`SNNRadiusBuilder`, a ``GraphBuilderCSR`` subclass that constructs the
generic **fixed-radius** spatial graph using SNN's fast, exact radius search,
and a thin :func:`spatial_neighbors` convenience wrapper.

Example
-------
>>> import squidpy as sq
>>> from snn_numba.squidpy import SNNRadiusBuilder
>>> sq.gr.spatial_neighbors_from_builder(adata, SNNRadiusBuilder(radius=50.0))

or, equivalently::

>>> from snn_numba.squidpy import spatial_neighbors
>>> spatial_neighbors(adata, radius=50.0)

This module imports squidpy at import time; it is an optional dependency.
Install it with ``uv sync --extra squidpy`` (or ``pip install squidpy``).  A
squidpy version exposing the builder API (``GraphBuilderCSR`` and
``spatial_neighbors_from_builder``) is required.
"""

from __future__ import annotations

import numpy as np

from ._graph import snn_radius_graph

# --- locate squidpy's builder base class (the public path moved over time) ---
try:
    from squidpy.gr.neighbors import GraphBuilderCSR
except ImportError:  # pragma: no cover - exercised only without the new API
    try:
        from squidpy.gr._build import GraphBuilderCSR  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "snn_numba.squidpy requires squidpy with the builder API "
            "(GraphBuilderCSR / spatial_neighbors_from_builder). Install a "
            "recent squidpy, e.g. `uv sync --extra squidpy` or "
            "`pip install squidpy`."
        ) from exc


class SNNRadiusBuilder(GraphBuilderCSR):
    """Fixed-radius spatial-graph builder backed by SNN.

    Connects every observation to all others within Euclidean distance
    ``radius`` of it -- squidpy's *generic radius graph* -- computed with the
    exact SNN sorting-based search instead of a KD-tree / brute force.

    Parameters
    ----------
    radius : float
        Connectivity radius in the coordinate units of ``adata.obsm[spatial_key]``.
    dtype : {numpy.float32, numpy.float64}, default numpy.float64
        Working precision of the SNN search. ``float32`` is faster; ``float64``
        reproduces a brute-force radius graph exactly.
    **kwargs
        Forwarded to ``GraphBuilderCSR`` (e.g. ``set_diag``, ``postprocessors``,
        ``library_key`` support handled by the base class).

    Notes
    -----
    A radius graph is symmetric, so the resulting connectivity matrix is
    symmetric and no extra symmetrisation step is needed.
    """

    def __init__(self, radius: float, *, dtype=np.float64, **kwargs):
        super().__init__(**kwargs)
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.radius = float(radius)
        self.dtype = np.dtype(dtype)

    def build_graph(self, coords):
        """Return ``(adj, dst)`` CSR matrices for the radius graph of ``coords``."""
        return snn_radius_graph(
            coords, radius=self.radius, set_diag=self.set_diag, dtype=self.dtype
        )

    def uns_params(self):
        """Configuration stored in ``adata.uns[key_added]`` by squidpy."""
        return {
            "radius": self.radius,
            "set_diag": self.set_diag,
            "backend": "snn_numba",
        }


def spatial_neighbors(adata, radius, *, dtype=np.float64, **kwargs):
    """Build an SNN radius graph on ``adata`` via squidpy's builder API.

    Thin wrapper around ``squidpy.gr.spatial_neighbors_from_builder`` using
    :class:`SNNRadiusBuilder`.

    Parameters
    ----------
    adata : AnnData | SpatialData
        Object with spatial coordinates in ``adata.obsm[spatial_key]``.
    radius : float
        Connectivity radius.
    dtype : {numpy.float32, numpy.float64}, default numpy.float64
        SNN working precision.
    **kwargs
        Split between the builder (``set_diag``, ``postprocessors``, …) and
        ``spatial_neighbors_from_builder`` (``spatial_key``, ``library_key``,
        ``key_added``, ``copy``, …).  Builder-only keys are forwarded to
        :class:`SNNRadiusBuilder`.

    Returns
    -------
    Whatever ``squidpy.gr.spatial_neighbors_from_builder`` returns (``None`` or
    a result object when ``copy=True``).
    """
    import squidpy as sq

    # keys consumed by spatial_neighbors_from_builder itself
    run_keys = {
        "spatial_key",
        "elements_to_coordinate_systems",
        "table_key",
        "library_key",
        "key_added",
        "copy",
    }
    run_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in run_keys}
    builder = SNNRadiusBuilder(radius, dtype=dtype, **kwargs)
    return sq.gr.spatial_neighbors_from_builder(adata, builder, **run_kwargs)


__all__ = ["SNNRadiusBuilder", "spatial_neighbors"]
