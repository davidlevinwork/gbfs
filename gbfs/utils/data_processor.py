from typing import Dict, Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from gbfs.models.data_view import DataCollection, DataProps, DataView

EPSILON = 1e-10


class DataProcessor:
    """
    The DataProcessor class is responsible for reading a dataset from a path,
    normalizing the data, and creating DataView objects with the dataset
    properties.
    """

    def __init__(self, dataset_path: str, label_column: str = 'class', cost_column: Optional[str] = None):
        """
        Initializes the DataProcessor with the dataset path and optional column names for label and cost.

        :param dataset_path: The file path to the dataset (CSV file expected).
        :param label_column: The name of the column to use as the label.
        :param cost_column: The name of the column to use as the feature cost (optional).
        """
        self.dataset_path = dataset_path
        self.label_column = label_column
        self.cost_column = cost_column

    def run(self) -> DataView:
        """
        Executes the data processing pipeline: reading the data, normalizing it,
        and creating a DataView object.

        :return: A DataView object with the original and normalized data and data properties.
        """
        data = pd.read_csv(self.dataset_path)
        norm_data = self._normalize_data(data)

        data_collections = self._create_data_collections(data, norm_data)
        data_props = self._compute_data_properties(data_collections['original'])

        return DataView(data=data_collections['original'],
                        norm_data=data_collections['normalized'],
                        data_props=data_props)

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the data using Min-Max scaling.

        :param data: The data to normalize.
        :return: The normalized data as a pandas DataFrame.
        """
        scaler = MinMaxScaler()
        cols_to_normalize = data.columns.difference([self.label_column])
        data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize] + EPSILON)
        return data

    def _create_data_collections(self, original_data: pd.DataFrame, normalized_data: pd.DataFrame) -> Dict[
        str, DataCollection]:
        """
        Creates collections for the original and normalized data.

        :param original_data: The original data as a pandas DataFrame.
        :param normalized_data: The normalized data as a pandas DataFrame.
        :return: A dictionary with keys 'original' and 'normalized' mapping to their respective DataCollection instances.
        """
        return {
            'original': DataCollection(x=original_data.drop(self.label_column, axis=1),
                                       y=pd.DataFrame(original_data[self.label_column]),
                                       x_y=original_data),
            'normalized': DataCollection(x=normalized_data.drop(self.label_column, axis=1),
                                         y=pd.DataFrame(normalized_data[self.label_column]),
                                         x_y=normalized_data)
        }

    def _compute_data_properties(self, data: DataCollection) -> DataProps:
        """
        Computes the data properties, such as label count and feature count.

        :param data: A DataCollection instance of the original data.
        :return: A DataProps instance with the computed properties.
        """
        labels = data.y[self.label_column].unique()
        features = data.x.columns
        feature_costs = self._compute_feature_costs(data.x)

        return DataProps(labels=labels,
                         n_labels=len(labels),
                         features=features,
                         n_features=len(features),
                         feature_costs=feature_costs)

    def _compute_feature_costs(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Computes the costs associated with each feature.

        :param features: A pandas DataFrame containing the feature data.
        :return: A dictionary mapping each feature to its cost.
        """
        if self.cost_column:
            try:
                costs = features[self.cost_column].fillna(EPSILON).tolist()
                return {feature: cost for feature, cost in zip(features.columns, costs)}
            except KeyError:
                raise KeyError(f"Cost column '{self.cost_column}' does not exist in the data.")
        else:
            return {feature: 1.0 for feature in features.columns}
