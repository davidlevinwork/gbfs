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
    cluster_labels = clustering['labels']
    cluster_centers = clustering['medoid_loc']

    intra_cluster_distances = euclidean_distances(
        space, cluster_centers[cluster_labels]
    ).diagonal()
    nearest_cluster_distances = np.zeros_like(intra_cluster_distances)

    for cluster_index, _ in enumerate(cluster_centers):
        cluster_members = space[cluster_labels == cluster_index]
        if cluster_members.size == 0:
            continue

        non_current_centers = np.delete(cluster_centers, cluster_index, axis=0)
        distances_to_non_current_centers = euclidean_distances(
            cluster_members, non_current_centers
        )
        nearest_cluster_distances[cluster_labels == cluster_index] = (
            distances_to_non_current_centers.mean(axis=1)
        )

    valid_distances_mask = intra_cluster_distances != 0
    if not np.any(valid_distances_mask):
        return 1

    a = intra_cluster_distances[valid_distances_mask]
    b = nearest_cluster_distances[valid_distances_mask]

    silhouette_scores = (b - a) / np.maximum(a, b)
    return np.mean(silhouette_scores)
