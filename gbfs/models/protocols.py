import numpy as np
from typing import Protocol, Any


class SupportsFitTransform(Protocol):
    """
    Protocol for dimensionality reduction models that support fit_transform method.

    This protocol expects that implementing models have a fit_transform method,
    which is a common interface for dimensionality reduction techniques in machine learning.
    """
    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        """
        :param X: The high-dimensional data to be reduced.
        :param y: Optional. Target variable (if required by the fit_transform method).
        :return: The dimensionality reduced data as a numpy ndarray.
        """
        ...
