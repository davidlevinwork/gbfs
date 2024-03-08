from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def calc_mss_value(space: np.ndarray, clustering: dict) -> Optional[float]:
    """
    Computes the mean simplified silhouette score value.

    :param space: A numpy array where each row represents a data point in the feature space.
    :param clustering: A dictionary containing the clustering information. It should have the
                       following keys:
                       - 'labels': an array where each element is the cluster label of the corresponding
                                   data point in 'space'.
                       - 'medoid_loc': a numpy array where each row is the centroid of a cluster in the
                                       feature space.

    :return: The mean silhouette score as a float. Higher scores indicate better clustering.
    """
    try:
        labels = clustering['labels']
        centroids = clustering['medoid_loc']

        a = euclidean_distances(space, centroids[labels]).diagonal()
        b = np.empty_like(a)

        for idx in range(len(centroids)):
            not_x_centroid = np.delete(centroids, idx, axis=0)
            distances_to_other_centroids = euclidean_distances(
                space[labels == idx], not_x_centroid
            )
            b[labels == idx] = distances_to_other_centroids.mean(axis=1)

        mask = a != 0
        a = a[mask]
        b = b[mask]

        sil_values = (b - a) / np.maximum(a, b)
        return np.mean(sil_values)
    except Exception as e:
        print(f'==> We have problem with value k={len(centroids)}, error: {e}. <==')
        return None
