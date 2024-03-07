import numpy as np


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

    var_p, var_q = std_p**2, std_q**2
    b = (1 / 8) * ((mean_p - mean_q) ** 2) * (2 / (var_p + var_q)) + 0.5 * np.log(
        (var_p + var_q) / (2 * (std_p * std_q))
    )
    return b
