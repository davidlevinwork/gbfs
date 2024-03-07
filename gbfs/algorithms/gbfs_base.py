from abc import ABC, abstractmethod


class GBFSBase(ABC):
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

    def plot_features(self):
        # implement here
        pass

    @property
    @abstractmethod
    def final_features(self):
        pass
