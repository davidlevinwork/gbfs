from itertools import combinations

import numpy as np

from gbfs.models.data_view import DataView, FeaturesGraph
from gbfs.utils.distance import get_classes, get_distance


class FeatureSpace:
    """
    The FeatureSpace class is responsible for computing a separability matrix for feature selection.

    :param data: DataView instance containing the dataset.
    :param separability_metric: The metric used to compute separability.
    :param dimension_reduction: The method used to reduce dimensionality.
    :param label_column: The name of the label column in the dataset.
    """

    def __init__(
        self,
        data: DataView,
        separability_metric: str,
        dimension_reduction: str,
        label_column: str,
    ) -> None:
        self.data = data
        self.separability_metric = separability_metric
        self.dimension_reduction = dimension_reduction
        self.label_column = label_column

    def run(self) -> FeaturesGraph:
        """
        Executes the process to compute the FeaturesGraph from the separability matrix.

        :return: FeaturesGraph object containing the separability matrix.
        """
        separability_matrix = self._compute_separability_matrix()
        return FeaturesGraph(
            sep_matrix=separability_matrix, reduced_sep_matrix=separability_matrix
        )

    def _compute_separability_matrix(self) -> np.ndarray:
        """
        Computes the separability matrix based on the defined metric.

        :return: Numpy ndarray representing the separability matrix.
        """
        label_combinations = list(combinations(np.unique(self.data.norm_data.y), 2))
        num_features = len(self.data.data_props.features)
        num_label_combinations = len(label_combinations)
        separation_matrix = np.zeros((num_features, num_label_combinations))

        for i, feature in enumerate(self.data.data_props.features):
            for j, labels in enumerate(label_combinations):
                class_1, class_2 = get_classes(
                    feature=feature,
                    label_1=labels[0],
                    label_2=labels[1],
                    data=self.data.norm_data.x_y,
                    label_column=self.label_column,
                )
                separation_matrix[i][j] = get_distance(
                    metric=self.separability_metric, c_1=class_1, c_2=class_2
                )

        return separation_matrix
