from itertools import combinations
import numpy as np
from gbfs.models.protocols import SupportsFitTransform

from gbfs.models.data_view import DataView, FeaturesGraph
from gbfs.utils.distance import get_distance


class FeatureSpace:
    """
    The FeatureSpace class is responsible for computing a separability matrix for feature selection.

    :param data: DataView instance containing the dataset.
    :param separability_metric: The metric used to compute separability.
    :param dim_reduction_model: The method used to reduce dimensionality.
    :param label_column: The name of the label column in the dataset.
    """

    def __init__(
        self,
        data: DataView,
        separability_metric: str,
        dim_reduction_model: SupportsFitTransform,
        label_column: str,
    ) -> None:
        self.data = data
        self.separability_metric = separability_metric
        self.dim_reduction_model = dim_reduction_model
        self.label_column = label_column

    def run(self) -> FeaturesGraph:
        """
        Executes the process to compute the FeaturesGraph from the separability matrix.

        :return: FeaturesGraph object containing the separability matrix.
        """
        separability_matrix = self._compute_separability_matrix()
        red_separability_matrix = self._reduce_dimension(separability_matrix)

        return FeaturesGraph(
            sep_matrix=separability_matrix, reduced_sep_matrix=red_separability_matrix
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
                label_1_values = self._extract_feature_values_by_class(
                    target_feature=feature,
                    class_label=labels[0]
                )
                label_2_values = self._extract_feature_values_by_class(
                    target_feature=feature,
                    class_label=labels[1]
                )
                separation_matrix[i][j] = get_distance(
                    metric=self.separability_metric, c_1=label_1_values, c_2=label_2_values
                )

        return separation_matrix

    def _reduce_dimension(self, data: np.ndarray) -> np.ndarray:
        reduced_data = self.dim_reduction_model.fit_transform(data)
        return reduced_data

    def _extract_feature_values_by_class(
            self,
            target_feature: str,
            class_label: str,
    ) -> np.ndarray:
        """
        Extracts values of a specified feature for two specific class labels from a dataset.

        :param target_feature: The feature for which values are to be extracted.
        :param class_label: The label of the class to filter the dataset.
        :return: NumPy array containing the feature values for the class.
        """
        class_values = self.data.norm_data.x_y.loc[self.data.norm_data.x_y[self.label_column] == class_label, target_feature].to_numpy()
        return class_values
