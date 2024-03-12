from unittest.mock import patch

from gbfs.algorithms.base import FeatureSelectorBase


@patch('gbfs.models.dim_reducer.DimReducerProtocol')
def test_initialization(mock_dim_reducer_protocol):
    """
    Test initialization of FeatureSelectorBase with mocked dependencies.
    """
    dataset_path = 'tests/algorithms/dataset.csv'
    separability_metric = 'jm'
    label_column = 'class'
    verbose = 1

    fs_base = FeatureSelectorBase(
        dataset_path, separability_metric, mock_dim_reducer_protocol, label_column, verbose
    )

    assert fs_base.verbose == verbose
    assert fs_base.dataset_path == dataset_path
    assert fs_base.label_column == label_column
    assert fs_base.dim_reducer_model is MockDimReducerProtocol
    assert fs_base.separability_metric == separability_metric
