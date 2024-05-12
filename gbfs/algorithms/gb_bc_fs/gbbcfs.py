from typing import Dict, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gbfs.algorithms.base import FeatureSelectorBase
from gbfs.algorithms.gb_bc_fs.heuristic import Heuristic
from gbfs.models.dim_reducer import DimReducerProtocol


class GBBCFS(FeatureSelectorBase):
    def __init__(
        self,
        dataset_path: str,
        separability_metric: str,
        dim_reducer_model: DimReducerProtocol,
        budget: int | float,
        label_column: str = 'class',
    ):
        super().__init__(
            dataset_path=dataset_path,
            separability_metric=separability_metric,
            dim_reducer_model=dim_reducer_model,
            label_column=label_column,
        )

        if budget <= 0:
            raise ValueError('Budget must be greater than 0.')
        self.budget = budget
        self.heuristic_result = None

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

        return self._execute_heuristic()

    def _execute_heuristic(self):
        """
        Executes the heuristic to determine the best (legal) feature selection within a specified budget.

        This function calculates the total cost of currently selected features and compares it with the available budget.
        If the total cost exceeds the budget, it initiates a Heuristic instance and runs it to find an optimal solution.
        Results from the heuristic run are printed. If the total cost is within the budget, the function simply returns.
        """
        selected_features_sum = self.get_selected_features_sum()

        if selected_features_sum <= self.budget:
            return self.selected_features.tolist()

        self.heuristic_result = Heuristic(
            data=self.data,
            budget=self.budget,
            knee_value=self.knee,
            clustering=self.clustering,
            feature_space=self.feature_space,
        ).run()
        return self.new_selected_features.tolist()

    def plot_feature_space(self):
        """
        Visualizes the feature space using scatter plots and highlights the selected features.
        Two plots are created side by side: one with original selected features, and one with updated selected features.
        The color intensity in the scatter plots represents the feature separability power, with a dedicated color-bar.
        """

        def plot_graph(ax, centroids, title):
            scatter = ax.scatter(
                feature_space[:, 0],
                feature_space[:, 1],
                c=c,
                cmap='viridis',
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5,
            )
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                marker='o',
                color='red',
                edgecolors='black',
                s=50,
                linewidth=1.5,
                label='Selected Features',
            )
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend()
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            return scatter

        sns.set(style='whitegrid')
        feature_space = self.feature_space.reduced_sep_matrix
        c = feature_space[:, 0] + feature_space[:, 1]

        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        cbar_ax = plt.subplot(gs[2])

        scatter1 = plot_graph(
            ax1,
            self.selected_features_loc,
            f'Original Result => k: {len(self.selected_features)} ; Cost: {self.cost:.3f} ; MSS: {self.mss:.3f}',
        )
        scatter2 = plot_graph(
            ax2,
            self.new_selected_features_loc,
            f'Updated Result => k: {len(self.new_selected_features)} ; Cost: {self.new_cost:.3f} ; MSS: {self.new_mss:.3f}',
        )

        fig.suptitle(
            'Feature Space with Selected Features Highlighted',
            fontsize=16,
            fontweight='bold',
        )
        plt.colorbar(scatter1, cax=cbar_ax, label='Feature Separability Power')

        plt.tight_layout()
        plt.show()

    def get_selected_features_sum(self) -> float:
        feature_costs = self.data.data_props.feature_costs
        return sum(
            feature_costs[key]
            for i, key in enumerate(feature_costs)
            if i in self.selected_features
        )

    @property
    def selected_features_to_cost(self) -> Dict[str, float]:
        feature_costs = self.data.data_props.feature_costs
        features_list = list(feature_costs.keys())

        indices = (
            self.new_selected_features
            if self.new_selected_features is not None
            else self.selected_features
        )
        return {features_list[i]: feature_costs[features_list[i]] for i in indices}

    @property
    def new_knee(self) -> Optional[int]:
        if self.heuristic_result:
            return len(self.heuristic_result['new_medoids'])
        return self.knee

    @property
    def new_mss(self) -> Optional[float]:
        if self.heuristic_result:
            return self.heuristic_result['mss']
        return self.mss

    @property
    def cost(self) -> Optional[float]:
        return self.get_selected_features_sum()

    @property
    def new_cost(self) -> Optional[float]:
        if self.heuristic_result:
            return self.heuristic_result['cost']
        return self.get_selected_features_sum()

    @property
    def new_selected_features(self) -> Optional[np.ndarray]:
        if self.heuristic_result:
            return self.heuristic_result['new_medoids']
        return self.selected_features

    @property
    def new_selected_features_loc(self) -> Optional[np.ndarray]:
        if self.heuristic_result:
            return self.heuristic_result['new_medoids_loc']
        return self.selected_features_loc
