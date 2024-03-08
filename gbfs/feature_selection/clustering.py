from gbfs.models.data_view import DataProps, FeaturesGraph
from sklearn_extra.cluster import KMedoids
import numpy as np
from gbfs.feature_selection.mss import calc_mss_value


class Clustering:
    """
    The Clustering class is designed to perform clustering on a reduced feature space
    to evaluate the clustering performance using different metrics.

    :param data_props: DataProps instance containing the dataset.
    :param feature_space: FeaturesGraph instance containing the reduced separability matrix.
    """

    def __init__(self, data_props: DataProps, feature_space: FeaturesGraph):
        self.data_props = data_props
        self.feature_space = feature_space

        self.features_len = len(self.data_props.features)

    def run(self):
        """
        Executes the clustering process in parallel for a range of k values and calculates the MSS (Mean Silhouette Score)
        for each clustering outcome.

        :return: A list of dictionaries, each containing the number of clusters (k), the MSS value, and
                 the kmedoids clustering result for that number of clusters.
        """
        results = []
        for k in range(2, self.features_len):
            kmedoids = self._run_kmedoids(data=self.feature_space.reduced_sep_matrix, k=k)
            mss = calc_mss_value(space=self.feature_space.reduced_sep_matrix, clustering=kmedoids)
            results.append({'k': k, 'mss': mss, 'kmedoids': kmedoids})

        return results

    @staticmethod
    def _run_kmedoids(data: np.ndarray, k: int) -> dict:
        """
        Performs KMedoids clustering on the provided data with a specified number of clusters (k).

        :param data: The dataset to cluster, as a NumPy ndarray.
        :param k: The number of clusters to use for KMedoids clustering.
        :return: A dictionary containing the cluster labels, indices of the medoids, and the locations of the medoids.
        """
        kmedoids = KMedoids(init='k-medoids++', n_clusters=k, method='pam').fit(data)

        return {
            'labels': kmedoids.labels_,
            'medoids': kmedoids.medoid_indices_,
            'medoid_loc': kmedoids.cluster_centers_
        }
