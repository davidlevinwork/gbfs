from typing import Dict, Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from gbfs.models.data_view import DataCollection, DataProps, DataView

EPSILON = 1e-10


class DataProcessor:
    """
    The DataProcessor class handles the preprocessing of data for feature selection algorithms.
    It reads data from a specified path, performs normalization using Min-Max scaling,
    and constructs DataView objects that encapsulate the dataset and its properties.

    This class is essential for preparing data before applying feature selection, ensuring
    that the data is in a suitable format and scale for analysis.

    :param dataset_path: Path to the dataset file (CSV format expected).
    :param label_column: Name of the column in the dataset that serves as the label. Defaults to 'class'.
    :param cost_column: Optional; name of the column that indicates the cost associated with each feature.
                        If not provided, all features are assumed to have equal cost.
    """

    def __init__(
        self,
        dataset_path: str,
        label_column: str = 'class',
        cost_column: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.label_column = label_column
        self.cost_column = cost_column

    def run(self) -> DataView:
        """
        Executes the data processing steps: loading the dataset, normalizing the feature values,
        and creating a DataView object containing the processed data along with its properties.

        :return: DataView object encapsulating the original and normalized datasets,
                 along with metadata about the dataset's properties.
        """
        data = pd.read_csv(self.dataset_path)
        norm_data = self._normalize_data(data)

        data_collections = self._create_data_collections(data, norm_data)
        data_props = self._compute_data_properties(data_collections['original'])

        return DataView(
            data=data_collections['original'],
            norm_data=data_collections['normalized'],
            data_props=data_props,
        )

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the data using Min-Max scaling.

        :param data: The data to normalize.
        :return: The normalized data as a pandas DataFrame.
        """
        scaler = MinMaxScaler()
        cols_to_normalize = data.columns.difference([self.label_column])
        data[cols_to_normalize] = scaler.fit_transform(
            data[cols_to_normalize] + EPSILON
        )
        return data

    def _create_data_collections(
        self, original_data: pd.DataFrame, normalized_data: pd.DataFrame
    ) -> Dict[str, DataCollection]:
        """
        Separates the original and normalized datasets into DataCollection instances,
        facilitating access to features, labels, and the combined dataset.

        :param original_data: The original dataset as a pandas DataFrame.
        :param normalized_data: The normalized dataset as a pandas DataFrame.
        :return: A dictionary containing 'original' and 'normalized' keys,
                 each mapped to their respective DataCollection instances.
        """
        return {
            'original': DataCollection(
                x=original_data.drop(self.label_column, axis=1),
                y=pd.DataFrame(original_data[self.label_column]),
                x_y=original_data,
            ),
            'normalized': DataCollection(
                x=normalized_data.drop(self.label_column, axis=1),
                y=pd.DataFrame(normalized_data[self.label_column]),
                x_y=normalized_data,
            ),
        }

    def _compute_data_properties(self, data: DataCollection) -> DataProps:
        """
        Computes and encapsulates the dataset's properties, such as the number of features,
        the number of labels, and optionally the cost associated with each feature.

        :param data: A DataCollection instance of the dataset.
        :return: A DataProps instance containing the dataset's properties.
        """
        labels = data.y[self.label_column].unique()
        features = data.x.columns
        feature_costs = self._compute_feature_costs(data.x)

        return DataProps(
            labels=labels,
            n_labels=len(labels),
            features=features,
            n_features=len(features),
            feature_costs=feature_costs,
        )

    def _compute_feature_costs(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Computes the cost associated with each feature, based on the values in the cost column,
        if such a column is specified. Otherwise, it defaults to a cost of 1.0 for all features.

        :param features: A pandas DataFrame of the features from the dataset.
        :return: A dictionary mapping each feature name to its associated cost.
        """
        if self.cost_column:
            try:
                costs = features[self.cost_column].fillna(EPSILON).tolist()
                return dict(zip(features.columns, costs))
            except KeyError:
                raise KeyError(
                    f"Cost column '{self.cost_column}' does not exist in the data."
                )
        else:
            return {feature: 1.0 for feature in features.columns}
