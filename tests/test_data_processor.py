from unittest.mock import MagicMock, patch

import pandas as pd

from gbfs.models.data_view import DataCollection, DataProps, DataView
from gbfs.utils.data_processor import DataProcessor


class MockMinMaxScaler:
    def __init__(self):
        self.fit_transform = MagicMock(side_effect=self.mocked_fit_transform)

    @staticmethod
    def mocked_fit_transform(data):
        return data


@patch('gbfs.utils.data_processor.pd.read_csv')
@patch('gbfs.utils.data_processor.MinMaxScaler', return_value=MockMinMaxScaler())
def test_data_processor_run(mock_min_max_scaler, mock_read_csv):
    """
    Test the run method of DataProcessor.

    This test ensures that the DataProcessor reads the dataset from a file,
    normalizes it using Min-Max scaling, and returns a DataView object with
    the expected data collections and properties.
    """
    mock_data = pd.DataFrame(
        {'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'class': [0, 1, 0]}
    )
    mock_read_csv.return_value = mock_data

    data_processor = DataProcessor('tests/dataset.csv')

    data_view = data_processor.run()

    mock_min_max_scaler.return_value.fit_transform.assert_called_once()
    assert isinstance(
        data_view, DataView
    ), 'The run method should return a DataView object.'


def test_compute_data_properties():
    """
    Test the _compute_data_properties method of DataProcessor.

    This test ensures that the DataProcessor correctly computes and encapsulates
    the dataset's properties such as the number of features, number of labels,
    and feature costs.
    """
    mock_data_collection = DataCollection(
        x=pd.DataFrame({'feature1': [0.1, 0.2, 0.3], 'feature2': [0.4, 0.5, 0.6]}),
        y=pd.DataFrame({'class': [0, 1, 0]}),
        x_y=pd.DataFrame(
            {
                'feature1': [0.1, 0.2, 0.3],
                'feature2': [0.4, 0.5, 0.6],
                'class': [0, 1, 0],
            }
        ),
    )
    feature_costs = {'feature1': 1.0, 'feature2': 1.0}

    data_processor = DataProcessor('tests/dataset.csv')

    data_props = data_processor._compute_data_properties(
        mock_data_collection, feature_costs
    )

    assert isinstance(data_props, DataProps), 'Should return a DataProps instance.'
    assert data_props.n_features == 2, 'There should be two features.'
    assert data_props.n_labels == 2, 'There should be two unique labels.'
    assert data_props.feature_costs == {
        'feature1': 1.0,
        'feature2': 1.0,
    }, 'Feature costs should be 1.0.'


def test_create_data_collections():
    """
    Test the _create_data_collections method of DataProcessor.

    This test checks that the DataProcessor correctly separates the original and
    normalized datasets into DataCollection instances, allowing for correct access
    to features, labels, and the full dataset.
    """
    original_data = pd.DataFrame(
        {'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'class': [0, 1, 0]}
    )
    normalized_data = pd.DataFrame(
        {'feature1': [0.0, 0.5, 1.0], 'feature2': [0.0, 0.5, 1.0], 'class': [0, 1, 0]}
    )

    data_processor = DataProcessor('tests/dataset.csv')

    data_collections = data_processor._create_data_collections(
        original_data, normalized_data
    )

    assert isinstance(data_collections, dict), 'Should return a dictionary.'
    assert 'original' in data_collections, "Dictionary should have an 'original' key."
    assert (
        'normalized' in data_collections
    ), "Dictionary should have a 'normalized' key."
    assert data_collections['original'].x.equals(
        original_data.drop('class', axis=1)
    ), 'Original features should match.'
    assert data_collections['original'].y.equals(
        pd.DataFrame(original_data['class'])
    ), 'Original labels should match.'
    assert data_collections['normalized'].x.equals(
        normalized_data.drop('class', axis=1)
    ), 'Normalized features should match.'
    assert data_collections['normalized'].y.equals(
        pd.DataFrame(normalized_data['class'])
    ), 'Normalized labels should match.'
