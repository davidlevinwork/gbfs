import unittest

import numpy as np

from gbfs.utils.distance import get_distance


class TestDistanceMetrics(unittest.TestCase):
    """
    Test suite for the distance metrics in the distance.py module.
    """

    def setUp(self):
        """
        Set up predefined data points for the distance metrics.
        """
        self.data_points_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.data_points_2 = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        self.data_points_3 = np.array([1.5, 1.6, 1.7, 1.8, 1.9])
        self.data_points_4 = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

        self.expected_bhattacharyya_1 = 0.9999999999999997
        self.expected_bhattacharyya_2 = 2.25

        self.expected_jm_1 = 1.264241117657115
        self.expected_jm_2 = 1.7892015508762713

        self.expected_wasserstein_1 = 0.4
        self.expected_wasserstein_2 = 0.6

    def test_bhattacharyya_distance(self):
        """
        Test the Bhattacharyya distance calculation against hard-coded expected values.
        """
        calculated_distance_1 = get_distance(
            'bhattacharyya', self.data_points_1, self.data_points_2
        )
        self.assertAlmostEqual(
            self.expected_bhattacharyya_1,
            calculated_distance_1,
            places=5,
            msg='Bhattacharyya distance calculation does not match the expected result.',
        )

        calculated_distance_2 = get_distance(
            'bhattacharyya', self.data_points_3, self.data_points_4
        )
        self.assertAlmostEqual(
            self.expected_bhattacharyya_2,
            calculated_distance_2,
            places=5,
            msg='Bhattacharyya distance calculation does not match the expected result.',
        )

    def test_jm_distance(self):
        """
        Test the Jeffries-Matusita distance calculation against hard-coded expected values.
        """
        calculated_distance_1 = get_distance(
            'jm', self.data_points_1, self.data_points_2
        )
        self.assertAlmostEqual(
            self.expected_jm_1,
            calculated_distance_1,
            places=5,
            msg='Jeffries-Matusita distance calculation does not match the expected result.',
        )

        calculated_distance_2 = get_distance(
            'jm', self.data_points_3, self.data_points_4
        )
        self.assertAlmostEqual(
            self.expected_jm_2,
            calculated_distance_2,
            places=5,
            msg='Jeffries-Matusita distance calculation does not match the expected result.',
        )

    def test_wasserstein_distance(self):
        """
        Test the Wasserstein distance calculation against values calculated using scipy.
        """
        calculated_distance_1 = get_distance(
            'wasserstein', self.data_points_1, self.data_points_2
        )
        self.assertAlmostEqual(
            self.expected_wasserstein_1,
            calculated_distance_1,
            places=5,
            msg='Wasserstein distance calculation does not match the expected result.',
        )

        calculated_distance_2 = get_distance(
            'wasserstein', self.data_points_3, self.data_points_4
        )
        self.assertAlmostEqual(
            self.expected_wasserstein_2,
            calculated_distance_2,
            places=5,
            msg='Wasserstein distance calculation does not match the expected result.',
        )

    def test_unsupported_metric(self):
        """
        Test the behavior when an unsupported metric name is provided.
        """
        with self.assertRaises(ValueError):
            get_distance('unsupported_metric', self.data_points_1, self.data_points_2)
