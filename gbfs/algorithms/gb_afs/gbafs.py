from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

from gbfs.algorithms.base import FeatureSelectorBase
from gbfs.models.dim_reducer import DimReducerProtocol


class GBAFS(FeatureSelectorBase):
    """
    Implements the Graph-Based Automatic Feature Selection (GB-AFS algorithm).

    GB-AFS is a novel graph-based filter method fpr automatic feature selection for multi-class classification tasks.
    The method determines the minimum combination of features required to sustain prediction performance ; It does not
    require any user-defined parameters such as the number of features to select.

    GB-AFS is agnostic to any combination of separability metric and dimensionality reduction technique, and the
    user can choose the desired combination.

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
        super().__init__(
            dataset_path=dataset_path,
            separability_metric=separability_metric,
            dim_reducer_model=dim_reducer_model,
            label_column=label_column,
        )

    def select_features(self) -> Optional[list]:
        """
        Executes the feature selection process by creating the feature space, evaluating clustering,
        finding the knee point, and finally selecting the features based on the clustering results.

        :return: A list of selected feature indices or None if no features are selected.
        """
        self._create_feature_space()
        self._evaluate_clustering()
        self._find_knee_point()
        self._find_features()

        return self.selected_features.tolist()

    def plot_feature_space(self):
        """
        Visualizes the feature space using a scatter plot and highlights the selected features.

        Centroids of selected features are marked distinctly to distinguish them from
        the rest of the features in the feature space. The color intensity in the scatter plot
        represents the feature separability power, with a color-bar for reference.
        """
        centroids = self.selected_features_loc
        feature_space = self.feature_space.reduced_sep_matrix

        sns.set(style='whitegrid')

        fig, ax = plt.subplots(figsize=(10, 8))
        c = feature_space[:, 0] + feature_space[:, 1]
        scatter = ax.scatter(
            feature_space[:, 0],
            feature_space[:, 1],
            c=c,
            cmap='viridis',
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5,
        )
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker='o',
            color='red',
            edgecolors='black',
            s=50,
            linewidth=1.5,
            label='Selected Features',
        )

        plt.colorbar(scatter, ax=ax, label='Feature Separability Power')

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.legend()
        plt.title('Feature Space with Selected Features Highlighted')
        plt.tight_layout()
        plt.show()
