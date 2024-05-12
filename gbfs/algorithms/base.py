from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from gbfs.feature_selection.clustering import Clustering
from gbfs.feature_selection.feature_space import FeatureSpace
from gbfs.feature_selection.knee_locator import KneesLocator
from gbfs.models.dim_reducer import DimReducerProtocol
from gbfs.utils.data_processor import DataProcessor


class FeatureSelectorBase(ABC):
    """
    Base class for feature selection using dimensionality reduction, separability metrics,
    and clustering to identify the most significant features in a dataset.

    The class processes the input dataset, evaluates feature space using the specified
    dimensionality reduction model and separability metric, performs clustering to find
    optimal feature subsets, and uses a knee detection method to select the number of
    features.

    :param dataset_path: Path to the dataset file.
    :param separability_metric: Metric used to evaluate separability of features.
    :param dim_reducer_model: Dimensionality reduction model to apply on the dataset.
    :param label_column: Name of the column in the dataset that contains the labels. Defaults to 'class'.
    """

    def __init__(
        self,
        dataset_path: str,
        separability_metric: str,
        dim_reducer_model: DimReducerProtocol,
        label_column: str = 'class',
    ):
        self.dataset_path = dataset_path
        self.label_column = label_column
        self.dim_reducer_model = dim_reducer_model
        self.separability_metric = separability_metric

        self.__process_data()

    @abstractmethod
    def select_features(self) -> Optional[list]:
        """
        Abstract method to execute the feature selection process.
        """

    def __process_data(self):
        """
        Processes the input dataset to prepare it for the feature selection process.
        """
        processor = DataProcessor(
            dataset_path=self.dataset_path,
            label_column=self.label_column,
        )
        self.data = processor.run()

    def _create_feature_space(self):
        """
        Creates the feature space using the provided dimensionality reduction model and separability metric.
        """
        feature_space = FeatureSpace(
            data=self.data,
            separability_metric=self.separability_metric,
            dim_reduction_model=self.dim_reducer_model,
            label_column=self.label_column,
        )
        self.feature_space = feature_space.run()

    def _evaluate_clustering(self):
        """
        Evaluates the clustering on the feature space to determine the clustering metrics for different numbers of clusters.
        """
        clustering = Clustering(
            data_props=self.data.data_props, feature_space=self.feature_space
        )
        self.clustering = clustering.run()

    def _find_knee_point(self):
        """
        Finds the knee point in the clustering metrics to determine the optimal number of features.
        """
        k_values = [x['k'] for x in self.clustering]
        mss_values = [x['mss'] for x in self.clustering]

        knee_locator = KneesLocator(x=k_values, y=mss_values)
        self._knee_locator = knee_locator.run()

    def _find_features(self):
        """
        Selects the features based on the clustering results at the knee point.
        """
        k_clustering = next(x for x in self.clustering if x['k'] == self.knee)

        self._mss_knee = k_clustering['mss']
        self._medoids = k_clustering['kmedoids']['medoids']
        self._medoids_loc = k_clustering['kmedoids']['medoid_loc']
        self._cluster_labels = k_clustering['kmedoids']['labels']

    @property
    def knee(self) -> Optional[int]:
        return self._knee_locator.knee

    @property
    def norm_knee(self) -> Optional[int]:
        return self._knee_locator.norm_knee

    @property
    def mss(self) -> Optional[float]:
        return self._mss_knee

    @property
    def knee_y(self) -> int | float | None:
        return self._knee_locator.knee_y

    @property
    def norm_knee_y(self) -> int | float | None:
        return self._knee_locator.norm_knee_y

    @property
    def number_of_features(self) -> Optional[int]:
        # add 2 because we don't check the first and last case
        return self._knee_locator.N + 2

    @property
    def selected_features(self) -> Optional[np.ndarray]:
        return self._medoids

    @property
    def selected_features_loc(self) -> Optional[np.ndarray]:
        return self._medoids_loc

    @property
    def separation_matrix(self) -> Optional[np.ndarray]:
        return self.feature_space.sep_matrix

    @property
    def reduced_separation_matrix(self) -> Optional[np.ndarray]:
        return self.feature_space.reduced_sep_matrix
