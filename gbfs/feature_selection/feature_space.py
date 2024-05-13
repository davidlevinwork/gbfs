from itertools import combinations

import numpy as np
from tqdm import tqdm

from gbfs.models.data_view import DataView, FeaturesGraph
from gbfs.models.dim_reducer import DimReducerProtocol
from gbfs.utils.distance import get_distance

STAGE_NAME = 'Separability-Based Feature Space'


class FeatureSpace:
    """
    The FeatureSpace class is responsible for computing a separability matrix for feature selection.

    :param data: DataView instance containing the dataset.
    :param separability_metric: The metric used to compute separability between features.
    :param dim_reduction_model: The method used to reduce dimensionality of the separability matrix.
    :param label_column: The name of the label column in the dataset.
    """

    def __init__(
        self,
        data: DataView,
        separability_metric: str,
        dim_reduction_model: DimReducerProtocol,
        label_column: str,
    ):
        self.data = data
        self.separability_metric = separability_metric
        self.dim_reduction_model = dim_reduction_model
        self.label_column = label_column

    def run(self) -> FeaturesGraph:
        """
        Executes the process to compute the FeaturesGraph from the separability matrix.

        :return: FeaturesGraph object containing the separability matrix and its dimensionally reduced form.
        """
        separability_matrix = self._compute_separability_matrix()
        red_separability_matrix = self._reduce_dimension(separability_matrix)

        return FeaturesGraph(
            sep_matrix=separability_matrix, reduced_sep_matrix=red_separability_matrix
        )

    def _compute_separability_matrix(self) -> np.ndarray:
        """
        Computes the separability matrix based on the defined metric. The separability is calculated
        between all pairs of class labels for each feature, providing insight into which features are
        most effective at distinguishing between classes.

        :return: Numpy ndarray representing the separability matrix.
        """
        label_combinations = list(combinations(np.unique(self.data.norm_data.y), 2))
        num_features = len(self.data.data_props.features)
        num_label_combinations = len(label_combinations)
        separation_matrix = np.zeros((num_features, num_label_combinations))

        for i, feature in tqdm(
            enumerate(self.data.data_props.features),
            total=num_features,
            desc=STAGE_NAME,
        ):
            for j, labels in enumerate(label_combinations):
                label_1_values = self._extract_feature_values_for_class(
                    feature_name=feature, class_name=labels[0]
                )
                label_2_values = self._extract_feature_values_for_class(
                    feature_name=feature, class_name=labels[1]
                )
                separation_matrix[i][j] = get_distance(
                    metric=self.separability_metric,
                    c_1=label_1_values,
                    c_2=label_2_values,
                )

        return separation_matrix

    def _reduce_dimension(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the dimensionality reduction model to the separability matrix, reducing its dimensions while attempting
        to preserve the significant separability features between class labels.

        :param data: The separability matrix as a NumPy ndarray.
        :return: The dimensionally reduced form of the separability matrix as a NumPy ndarray.
        """
        reduced_data = self.dim_reduction_model.fit_transform(data)
        return reduced_data

    def _extract_feature_values_for_class(
        self, feature_name: str, class_name: str
    ) -> np.ndarray:
        """
        Retrieves all values of a specified feature from the dataset, but only for rows that belong to a specific class.

        :param feature_name: Name of the feature for which values are to be extracted.
        :param class_name: Name of the class to use for filtering the dataset.
        :return: A NumPy array containing the extracted feature values for the specified class.
        """
        filtered_rows = self.data.norm_data.x_y[
            self.data.norm_data.x_y[self.label_column] == class_name
        ]
        feature_values = filtered_rows[feature_name].to_numpy()
        return feature_values
