import seaborn as sns
import matplotlib.pyplot as plt
from .base import FeatureSelectorBase
from gbfs.models.dim_reducer import DimReducerProtocol


class GBAFS(FeatureSelectorBase):
    def __init__(
        self,
        dataset_path: str,
        separability_metric: str,
        dim_reducer_model: DimReducerProtocol,
        label_column: str = 'class',
        verbose: int = 1,
    ):
        super().__init__(dataset_path=dataset_path, separability_metric=separability_metric, dim_reducer_model=dim_reducer_model, label_column=label_column, verbose=verbose)

    def select_features(self):
        return super().select_features()

    def plot_feature_space(self):
        centroids = self.selected_features_loc
        feature_space = self.feature_space.reduced_sep_matrix

        sns.set(style="whitegrid")

        fig, ax = plt.subplots(figsize=(10, 8))
        c = feature_space[:, 0] + feature_space[:, 1]
        scatter = ax.scatter(feature_space[:, 0], feature_space[:, 1], c=c, cmap='viridis', alpha=0.7, edgecolors='w',
                             linewidth=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='red', edgecolors='black', s=50, linewidth=1.5,
                    label="Selected Features")

        # Improve readability
        plt.colorbar(scatter, ax=ax, label='Feature Separability Power')

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.legend()
        plt.title("Feature Space with Selected Features Highlighted")
        plt.tight_layout()
        plt.show()
