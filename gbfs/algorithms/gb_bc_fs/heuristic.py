from typing import Optional, Tuple

import numpy as np
from scipy.spatial import distance
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from gbfs.feature_selection.mss import calc_mss_value
from gbfs.models.data_view import DataView, FeaturesGraph

STAGE_NAME = 'Heuristic Stage'


class Heuristic:
    def __init__(
        self,
        data: DataView,
        clustering: list,
        feature_space: FeaturesGraph,
        knee_value: int,
        budget: float,
        alpha: float = 0.5,
        epochs: int = 100,
    ):
        self.data = data
        self.budget = budget
        self.knee_value = knee_value
        self.clustering = clustering
        self.feature_space = feature_space
        self.feature_costs = self.data.data_props.feature_costs

        # heuristic properties
        self.alpha = alpha
        self.epochs = epochs
        self.is_heuristic_completed = False

    def run(self) -> Optional[dict]:
        """
        Executes the heuristic process. Iterates from 'knee_value' found down to 2 to find a subset of new valid
        features within the budget using a heuristic approach. For each 'k':
        - If the cost of the 'k' cheapest features exceeds the budget, it continues to the next k
          (there isn't a valid solution with k features)
        - If the cost of the 'k' selected features is within the budget, the function returns the solution.
        - Otherwise, it executes the heuristic to try to find a viable solution with the current 'k'.

        :return: A dictionary containing the selected features if a solution is found that fits within the budget.
        :raises RuntimeError: If the heuristic fails to find a solution within the predefined epochs for each 'k'.
        """
        for k in range(self.knee_value, 2, -1):
            min_k_features_cost = sum(sorted(self.feature_costs.values())[:k])
            selected_k_features_cost = self._get_selected_medoids_cost(k=k)

            if min_k_features_cost > self.budget:
                continue
            if selected_k_features_cost <= self.budget:
                return self._get_original_k_clustering_result(k=k)
            else:
                result = self._execute_heuristic(k=k)
                if result:
                    return result

        raise RuntimeError(
            f'Heuristic failed to find a solution within {self.epochs} epochs. '
            f'Consider increasing the number of epochs.'
        )

    def _execute_heuristic(self, k: int) -> Optional[dict]:
        """
        Performs a heuristic-based KMedoids clustering on the provided dataset with a specified number of clusters (k).

        This method tries to optimize the clustering process under a predefined budget constraint over several epochs.
        It dynamically adjusts the clustering configuration if the minimal clustering cost exceeds the allowed budget.

        :param k: The number of clusters to use for KMedoids clustering.
        :return: A dictionary containing the cluster labels, indices of the medoids, and the locations of the medoids.
        :raises RuntimeError: If the heuristic fails to find a solution within the predefined epochs.
        """
        current_kmedoids = next(
            (config['kmedoids'] for config in self.clustering if config['k'] == k), None
        )

        for _ in tqdm(
            range(self.epochs), total=self.epochs, desc=f'{STAGE_NAME} for k=[{k}]'
        ):
            clustering_cost = self._get_min_clustering_cost(clustering=current_kmedoids)

            if clustering_cost > self.budget:
                current_kmedoids = self._run_kmedoids(k=k)
                continue

            self._set_clustering_info(clustering=current_kmedoids)
            self._update_clustering()

            if self.is_heuristic_completed:
                return self._calculate_new_feature_space(k=k)

        return None

    def _set_clustering_info(self, clustering: dict):
        """
        Sets detailed information about the current clustering configuration,
        including costs and distances associated with medoids and their corresponding clusters.

        :param clustering: A dictionary containing clustering data with keys 'medoids', 'medoid_loc', and 'labels'.
        """
        clusters = []
        data_properties = self.data.data_props
        feature_names = list(data_properties.feature_costs.keys())
        feature_costs = list(data_properties.feature_costs.values())

        for medoid_index, medoid_location in zip(
            clustering['medoids'], clustering['medoid_loc']
        ):
            cluster_label = clustering['labels'][medoid_index]
            cluster_feature_indices = [
                index
                for index, label in enumerate(clustering['labels'])
                if label == cluster_label
            ]

            medoid_cost = feature_costs[medoid_index]
            medoid_name = feature_names[medoid_index]
            cluster_feature_names = [
                feature_names[index] for index in cluster_feature_indices
            ]
            cluster_feature_costs = [
                feature_costs[index] for index in cluster_feature_indices
            ]
            min_distance, max_distance = self._get_cluster_distance(
                medoid_index, cluster_feature_indices
            )
            total_cluster_cost = sum(cluster_feature_costs)

            clusters.append(
                {
                    'cluster_label': cluster_label,
                    'medoid': medoid_index,
                    'medoid_loc': medoid_location,
                    'medoid_cost': medoid_cost,
                    'medoid_name': medoid_name,
                    'cluster_features_idx': cluster_feature_indices,
                    'cluster_features_name': cluster_feature_names,
                    'cluster_features_cost': cluster_feature_costs,
                    'cluster_total_cost': total_cluster_cost,
                    'cluster_distances': {
                        'max_dist': max_distance,
                        'min_dist': min_distance,
                    },
                }
            )

        self.cluster_details = sorted(
            clusters, key=lambda cluster: cluster['medoid_cost'], reverse=True
        )

    def _get_cluster_distance(
        self, medoid_index: int, feature_indices: list
    ) -> Tuple[float, float]:
        """
        Calculates the minimum and maximum Euclidean distances from a given medoid to other features in the cluster.

        If the cluster contains only one feature, the function returns (0, 0) as no distance calculation is needed.
        Otherwise, it computes the Euclidean distance from the medoid to each feature in the cluster.

        :param medoid_index: The index of the medoid in the feature space.
        :param feature_indices: A list of indices for features that belong to the same cluster as the medoid.
        :return: A tuple containing the minimum and maximum distances from the medoid to the features in the cluster.
        """
        if len(feature_indices) == 1:
            # No need to calculate distances if there's only one feature
            return 0.0, 0.0

        space = self.feature_space.reduced_sep_matrix
        distances = np.array(
            [
                distance.euclidean(space[feature_idx], space[medoid_index])
                for feature_idx in feature_indices
            ]
        )
        return distances.min(), distances.max()

    def _update_clustering(self):
        """
        Updates the current clustering configuration based on budget constraints and potential improvements.

        This method iteratively checks each cluster against the total cost and budget constraints.
        If the total cost of medoids is under the budget, it completes the heuristic. For clusters with more than
        one feature, it attempts to optimize by possibly selecting a new medoid within a recalculated budget
        considering the surplus cost.

        The method updates the medoid if a more cost-effective option is found, enhancing overall clustering efficiency.
        """
        for idx, cluster in enumerate(self.cluster_details):
            total_medoid_cost = sum(
                cluster['medoid_cost'] for cluster in self.cluster_details
            )
            if total_medoid_cost <= self.budget:
                self.is_heuristic_completed = True
                return

            if len(cluster['cluster_features_idx']) == 1:
                continue  # Skip updates for clusters of size 1

            surplus_cost = self._calculate_surplus_cost(cluster_index=idx)
            adjusted_budget = self.budget - surplus_cost
            potential_new_medoid = self._select_new_medoid(
                cluster=cluster, budget=adjusted_budget
            )

            if potential_new_medoid[0] is None:
                continue  # No suitable new medoid found within the adjusted budget

            if potential_new_medoid[0] != cluster['medoid']:
                self._update_cluster_medoid(idx=idx, new_medoid=potential_new_medoid)

    def _calculate_surplus_cost(self, cluster_index: int) -> float:
        """
        Calculates the surplus cost of the clustering config excluding the current cluster at the specified index.

        This method computes the sum of the medoid costs of all clusters before the given index (backward cost) and
        the minimum feature costs of all clusters after the given index (forward cost), and returns the total cost.

        :param cluster_index: The index of the current cluster within the cluster details list.
        :return: The sum of backward and forward costs as a float, representing the surplus cost.
        """
        # Calculate backward cost: Sum of medoid costs for all clusters before the current index
        backward_cost = sum(
            item['medoid_cost'] for item in self.cluster_details[:cluster_index]
        )

        # Calculate forward cost: Sum of the minimum feature costs for all clusters after the current index
        forward_cost = sum(
            min(res['cluster_features_cost'])
            for res in self.cluster_details[cluster_index + 1 :]
        )

        return backward_cost + forward_cost

    def _select_new_medoid(
        self, cluster: dict, budget: float
    ) -> Tuple[Optional[int], float]:
        """
        Selects a new medoid for a cluster based on the best scoring feature within a specified budget.

        This method evaluates each feature within the cluster's feature costs that are under the budget.
        The feature with the lowest score, indicating better suitability as a medoid, is selected.

        :param cluster: A dictionary containing details about the cluster.
        :param budget: A float indicating the maximum allowable cost for the new medoid.
        :return: A tuple containing the index of the best new medoid and its cost, or (None, float('inf')) if no
                 suitable medoid is found.
        """
        best_score = float('inf')
        best_cost = float('inf')
        best_feature_idx = None

        is_special_distance = len(cluster['cluster_features_idx']) == 2

        relevant_features = [
            (idx, cost)
            for idx, cost in zip(
                cluster['cluster_features_idx'], cluster['cluster_features_cost']
            )
            if cost < budget
        ]

        for feature_idx, feature_cost in relevant_features:
            score = self._calculate_feature_score(
                cluster, (feature_idx, feature_cost), is_special_distance
            )
            if score < best_score:
                best_score, best_cost, best_feature_idx = (
                    score,
                    feature_cost,
                    feature_idx,
                )

        return best_feature_idx, best_cost

    def _calculate_feature_score(
        self,
        cluster: dict,
        feature: Tuple[int, float],
        is_special_distance: bool,
        epsilon: float = 1e-10,
    ) -> float:
        """
        Calculates a score for a feature based on its cost and distance to the medoid, adjusted by normalization and a
        weighting factor.

        The function calculates and normalizes the Euclidean distance and cost of a feature relative to the cluster's
        medoid and respective ranges.

        :param cluster: A dictionary containing details about the cluster.
        :param feature: A tuple containing the index and cost of the feature being scored.
        :param is_special_distance: A boolean indicating if a special distance calculation condition applies.
        :param epsilon: A small float added to denominators to prevent division by zero, defaulting to 1e-10.
        :return: A float representing the calculated score of the feature.
        """
        feature_idx, feature_cost = feature
        feature_to_medoid_dist = distance.euclidean(
            self.feature_space.reduced_sep_matrix[feature_idx],
            self.feature_space.reduced_sep_matrix[cluster['medoid']],
        )

        if feature_idx == cluster['medoid'] or is_special_distance:
            normalized_distance = 0
        else:
            distance_range = (
                cluster['cluster_distances']['max_dist']
                - cluster['cluster_distances']['min_dist']
            )
            normalized_distance = np.divide(
                feature_to_medoid_dist - cluster['cluster_distances']['min_dist'],
                distance_range + epsilon,
            )

        cost_range = max(cluster['cluster_features_cost']) - min(
            cluster['cluster_features_cost']
        )
        normalized_cost = np.divide(
            feature_cost - min(cluster['cluster_features_cost']), cost_range + epsilon
        )

        score = self.alpha * normalized_cost + (1 - self.alpha) * normalized_distance

        return score

    def _get_min_clustering_cost(self, clustering: dict) -> float:
        """
        Calculates the min total cost of a clustering configuration based on the lowest cost feature in each cluster.

        :param clustering: A dictionary containing 'labels' which map to the cluster labels of each feature.
        :return: The sum of the minimum costs for each cluster label, representing the minimal possible clustering cost.
        """
        min_costs = {}
        for feature, label in zip(self.feature_costs, clustering['labels']):
            cost = self.feature_costs[feature]
            # Update the minimum cost for each label
            if label not in min_costs or cost < min_costs[label]:
                min_costs[label] = cost

        return sum(min_costs.values())

    def _get_selected_medoids_cost(self, k: int) -> float:
        """
        Computes the total cost of medoids for the specified number of clusters (k).

        :param k: The number of clusters.
        :return: The total cost of medoids for the given number of clusters.
        """
        return sum(
            list(self.data.data_props.feature_costs.values())[feature]
            for cluster_info in self.clustering
            if cluster_info['k'] == k
            for feature in cluster_info['kmedoids']['medoids']
        )

    def _run_kmedoids(self, k: int) -> dict:
        """
        Performs KMedoids clustering on the provided data with a specified number of clusters (k).

        :param k: The number of clusters to use for KMedoids clustering.
        :return: A dictionary containing the cluster labels, indices of the medoids, and the locations of the medoids.
        """
        kmedoids = KMedoids(init='k-medoids++', n_clusters=k, method='pam').fit(
            self.feature_space.reduced_sep_matrix
        )

        return {
            'labels': kmedoids.labels_,
            'medoids': kmedoids.medoid_indices_,
            'medoid_loc': kmedoids.cluster_centers_,
        }

    def _update_cluster_medoid(self, idx: int, new_medoid: Tuple[int, float]):
        """
        Updates the medoid details in the cluster at the specified index.

        :param idx: Index of the cluster in the cluster details list.
        :param new_medoid: A tuple containing the index of the new medoid and its cost.
        """
        self.cluster_details[idx].update(
            {
                'medoid': new_medoid[0],
                'medoid_loc': self.feature_space.reduced_sep_matrix[new_medoid[0]],
                'medoid_cost': new_medoid[1],
                'medoid_name': self.data.data_props.features[new_medoid[0]],
            }
        )

    def _calculate_new_feature_space(self, k: int) -> dict:
        """
        Calculates the new feature space and evaluates it using a new set of kmedoids and the MSS.

        :param k: The number of clusters used to define the new kmedoids.
        :return: A dictionary with the MSS value, total cost, and details of the new medoids.
        """
        kmedoids = self._get_new_kmedoids()
        mss = calc_mss_value(
            space=self.feature_space.reduced_sep_matrix, clustering=kmedoids
        )
        cost = sum(cluster['medoid_cost'] for cluster in self.cluster_details)

        return {
            'mss': mss,
            'cost': cost,
            'new_labels': kmedoids['labels'],
            'new_medoids': kmedoids['medoids'],
            'new_medoids_loc': kmedoids['medoid_loc'],
        }

    def _get_new_kmedoids(self) -> dict:
        """
        Generates a new kmedoids clustering based on the current cluster details, after the heuristic has been applied
        and found a new feature space, based on new medoids.

        :return: A dictionary with new labels, medoid indices, and medoid locations.
        """
        medoids_idx = [medoid['medoid'] for medoid in self.cluster_details]
        medoids = [
            self.feature_space.reduced_sep_matrix[center] for center in medoids_idx
        ]
        labels = []

        for feature in self.feature_space.reduced_sep_matrix:
            closest_centroid_idx = np.argmin(
                [
                    distance.euclidean(
                        feature, self.feature_space.reduced_sep_matrix[medoid]
                    )
                    for medoid in medoids_idx
                ]
            )
            labels.append(closest_centroid_idx)

        return {
            'labels': np.array(labels),
            'medoids': np.array(medoids_idx),
            'medoid_loc': np.array(medoids),
        }

    def _get_original_k_clustering_result(self, k: int) -> dict:
        """
        Retrieves the original clustering results for a specified number of clusters 'k'.

        :param k: The number of clusters to retrieve the original clustering results for.
        :return: A dictionary containing the results of the solution.
        """
        current_kmedoids = next(
            (config['kmedoids'] for config in self.clustering if config['k'] == k), None
        )
        return {
            'mss': current_kmedoids['mss'],
            'cost': self._get_selected_medoids_cost(k=k),
            'new_labels': current_kmedoids['kmedoids']['labels'],
            'new_medoids': current_kmedoids['kmedoids']['medoids'],
            'new_medoids_loc': current_kmedoids['kmedoids']['medoid_loc'],
        }
