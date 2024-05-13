from typing import Optional
from unittest.mock import patch
from gbfs.algorithms.base import FeatureSelectorBase
import pandas as pd


class ConcreteFeatureSelector(FeatureSelectorBase):
    def select_features(self) -> Optional[list]:
        return []


@patch('gbfs.models.dim_reducer.DimReducerProtocol')
@patch('pandas.read_csv')
def test_initialization(mock_read_csv, mock_dim_reducer_protocol):
    """
    Test initialization of FeatureSelectorBase with mocked dependencies.
    """
    # Mock read_csv to return a DataFrame
    mock_read_csv.return_value = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'class': [0, 1, 0]
    })

    dataset_path = 'tests/dataset.csv'
    separability_metric = 'jm'
    label_column = 'class'

    fs_base = ConcreteFeatureSelector(
        dataset_path,
        separability_metric,
        mock_dim_reducer_protocol,
        label_column,
    )

    assert fs_base.dataset_path == dataset_path
    assert fs_base.label_column == label_column
    assert fs_base.dim_reducer_model is mock_dim_reducer_protocol
    assert fs_base.separability_metric == separability_metric
