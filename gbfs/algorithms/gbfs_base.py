from typing import Optional
from gbfs.utils.data_processor import DataProcessor
from gbfs.models.protocols import SupportsFitTransform
from gbfs.feature_selection.feature_space import FeatureSpace

from gbfs.models.data_view import DataView
from abc import ABC, abstractmethod


class GBFSBase(ABC):
    def __init__(
            self,
            dataset_path: str,
            separability_metric: str,
            dimension_reduction: SupportsFitTransform,
            label_column: str = 'class',
            verbose: int = 1,
    ):
        self.separability_metric = separability_metric
        self.dimension_reduction = dimension_reduction
        self.dataset_path = dataset_path
        self.verbose = verbose
        self.label_column = label_column

        self.data_view: Optional[DataView] = None
        self.feature_space: Optional[FeatureSpace] = None

    def find_features(self):
        self._process_data()
        self._compute_feature_space()

    def _process_data(self):
        processor = DataProcessor(
            dataset_path=self.dataset_path, label_column=self.label_column
        )
        self.data_view = processor.run()

    def _compute_feature_space(self):
        feature_space = FeatureSpace(
            data=self.data_view,
            separability_metric=self.separability_metric,
            dim_reduction_model=self.dimension_reduction,
            label_column=self.label_column,
        )
        self.feature_space = feature_space.run()

    def plot_features(self):
        # implement here
        pass

    @property
    def final_features(self):
        pass
