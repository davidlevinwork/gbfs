from unittest.mock import patch

from gbfs.algorithms.base import FeatureSelectorBase


@patch('gbfs.models.dim_reducer.DimReducerProtocol')
def test_initialization(mock_dim_reducer_protocol):
    """
    Test initialization of FeatureSelectorBase with mocked dependencies.
    """
    dataset_path = 'tests/dataset.csv'
    separability_metric = 'jm'
    label_column = 'class'

    fs_base = FeatureSelectorBase(
        dataset_path,
        separability_metric,
        mock_dim_reducer_protocol,
        label_column,
    )

    assert fs_base.dataset_path == dataset_path
    assert fs_base.label_column == label_column
    assert fs_base.dim_reducer_model is mock_dim_reducer_protocol
    assert fs_base.separability_metric == separability_metric
