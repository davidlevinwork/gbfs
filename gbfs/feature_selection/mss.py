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
    labels = clustering['labels']
    centroids = clustering['medoid_loc']

    # distance of each point to its own cluster centroid
    a = euclidean_distances(space, centroids[labels]).diagonal()
    b = np.zeros_like(a)

    for idx in range(len(centroids)):
        # list of all the other centroids (excluding the current cluster (idx))
        other_centroids = np.delete(centroids, idx, axis=0)

        # calculate the distance from all points in the current cluster (idx) to all other centroids
        distances_to_other_centroids = euclidean_distances(
            space[labels == idx], other_centroids
        )

        # store the mean distance of each point in the current cluster to the centroids of other clusters
        b[labels == idx] = distances_to_other_centroids.mean(axis=1)

    # bool mask to filter out points that have zero distance to their own centroid (1-point cluster) + apply mask
    mask = a != 0
    a = a[mask]
    b = b[mask]

    sil_values = (b - a) / np.maximum(a, b)
    return np.mean(sil_values)
