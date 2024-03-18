from typing import Any, Protocol

import numpy as np


class DimReducerProtocol(Protocol):
    """
    Protocol for dimensionality reduction models that support fit_transform method.

    This protocol expects that implementing models have a fit_transform method,
    which is a common interface for dimensionality reduction techniques in machine learning.
    """

    def fit_transform(self, x: np.ndarray, y: Any = None) -> np.ndarray:
        """
        :param x: The high-dimensional data to be reduced.
        :param y: Optional. Target variable (if required by the fit_transform method).
        :return: The dimensionality reduced data as a numpy ndarray.
        """
        ...
