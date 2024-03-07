from typing import Tuple

import numpy as np
import pandas as pd


def get_classes(data: pd.DataFrame, feature: str, label_1: str, label_2: str, label_column: str) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Retrieves classes based on specified feature and labels.

    :param data: DataFrame containing the dataset.
    :param feature: The feature for which the classes are to be retrieved.
    :param label_1: The first label for class comparison.
    :param label_2: The second label for class comparison.
    :param label_column: The name of the label column in the dataset.
    :return: Tuple of numpy ndarrays for the two classes.
    """
    c_1 = data.loc[data[label_column] == label_1, feature].to_numpy()
    c_2 = data.loc[data[label_column] == label_2, feature].to_numpy()
    return c_1, c_2


def get_distance(metric: str, c_1: np.ndarray, c_2: np.ndarray) -> float:
    """
    Computes the distance between two classes based on the specified metric.

    :param metric: The name of the metric to use for computing distance.
    :param c_1: Class 1 data points.
    :param c_2: Class 2 data points.
    :return: The computed distance metric.
    """
    if metric == 'jm':
        return jm_distance(c_1, c_2)
    else:
        raise ValueError(f"Unsupported metric '{metric}' provided.")


def jm_distance(p: np.ndarray, q: np.ndarray):
    b = bhattacharyya_distance(p, q)
    jm = 2 * (1 - np.exp(-b))
    return jm


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray):
    mean_p, mean_q = p.mean(), q.mean()
    std_p = p.std() if p.std() != 0 else 0.00000000001
    std_q = q.std() if q.std() != 0 else 0.00000000001

    var_p, var_q = std_p ** 2, std_q ** 2
    b = (1 / 8) * ((mean_p - mean_q) ** 2) * (2 / (var_p + var_q)) + \
        0.5 * np.log((var_p + var_q) / (2 * (std_p * std_q)))
    return b
