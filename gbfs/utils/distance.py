import numpy as np


def get_distance(metric: str, c_1: np.ndarray, c_2: np.ndarray) -> float:
    """
    Computes the distance between two classes based on the specified metric.
    This function serves as a dispatcher that selects the appropriate distance calculation
    method according to the metric specified.

    Currently, supports:
    - 'jm': Jeffries-Matusita (JM) distance
    - 'bhattacharyya': Bhattacharyya distance
    - 'wasserstein': Wasserstein distance

    :param metric: The name of the metric to use for computing the distance.
    :param c_1: Numpy array representing the data points of the first class.
    :param c_2: Numpy array representing the data points of the second class.
    :return: The computed distance metric as a float.
    :raises ValueError: If an unsupported metric name is provided.
    """
    if metric == 'jm':
        return jm_distance(c_1, c_2)
    elif metric == 'bhattacharyya':
        return bhattacharyya_distance(c_1, c_2)
    elif metric == 'wasserstein':
        from scipy.stats import wasserstein_distance

        return wasserstein_distance(c_1, c_2)
    else:
        raise ValueError(f"Unsupported metric '{metric}' provided.")


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) / 2)


def jm_distance(p: np.ndarray, q: np.ndarray) -> float:
    b = bhattacharyya_distance(p, q)
    jm = 2 * (1 - np.exp(-b))
    return jm


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    mean_p, mean_q = p.mean(), q.mean()
    std_p = p.std() if p.std() != 0 else 1e-10
    std_q = q.std() if q.std() != 0 else 1e-10

    var_p, var_q = std_p**2, std_q**2
    b = (1 / 8) * ((mean_p - mean_q) ** 2) * (2 / (var_p + var_q)) + 0.5 * np.log(
        (var_p + var_q) / (2 * (std_p * std_q))
    )
    return b
