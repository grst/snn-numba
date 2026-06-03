"""snn_numba: fast, exact, radius-based nearest-neighbor search in Python/Numba.

Example
-------
>>> import numpy as np
>>> from snn_numba import SNN
>>> X = np.random.default_rng(0).random((10000, 50))
>>> snn = SNN(X)                       # build the index
>>> ind = snn.query_radius(X[:1], 0.8) # neighbors within radius 0.8
"""

from .snn import SNN

__all__ = ["SNN"]
__version__ = "0.1.0"
