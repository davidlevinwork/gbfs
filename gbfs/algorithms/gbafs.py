from typing import Optional

from gbfs.feature_selection.feature_space import FeatureSpace
from gbfs.models.data_view import DataView
from gbfs.utils.data_processor import DataProcessor


class GBAFS:
    def __init__(
        self,
        dataset_path: str,
        separability_metric: str,
        dimension_reduction: str,
        label_column: str = 'class',
        verbose: int = 1,
        test_size: int = 0.25,
    ):
        self.separability_metric = separability_metric
        self.dimension_reduction = dimension_reduction
        self.dataset_path = dataset_path
        self.verbose = verbose
        self.label_column = label_column
        self.test_size = test_size

        self.data_view: Optional[DataView] = None

    def find_features(self):
        # 1: prepare data
        self._process_data()

        # 2: sep matrix
        self._compute_feature_space()
        # 2: create sep matrix + reduce dim (if dim is not 2, return error for the user)
        # 3: for each k run clustering eval: clustering + mss
        # 4: find knee based on mss graph
        # 5: return features based on the knee value
        x = 5

    def _process_data(self):
        processor = DataProcessor(
            dataset_path=self.dataset_path, label_column=self.label_column
        )
        self.data_view = processor.run()

    def _compute_feature_space(self):
        x = FeatureSpace(
            data=self.data_view,
            separability_metric=self.separability_metric,
            dimension_reduction=self.dimension_reduction,
            label_column=self.label_column,
        )
