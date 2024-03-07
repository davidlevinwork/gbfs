from gbfs.models.protocols import SupportsFitTransform
from .gbfs_base import GBFSBase


class GBAFS(GBFSBase):
    def __init__(
        self,
        dataset_path: str,
        separability_metric: str,
        dimension_reduction: SupportsFitTransform,
        label_column: str = 'class',
        verbose: int = 1,
    ):
        super().__init__(dataset_path, separability_metric, dimension_reduction, label_column, verbose)

    def find_features(self):
        super().find_features()

        # 1: prepare data
        # 2: sep matrix

        # 2: create sep matrix + reduce dim (if dim is not 2, return error for the user)
        # 3: for each k run clustering eval: clustering + mss
        # 4: find knee based on mss graph
        # 5: return features based on the knee value
        x = 5




