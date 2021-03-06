import unittest

import numpy as np

import cimem.core


class TestCluster(unittest.TestCase):
    """Test the cimem.core.Cluster class"""

    def test_init(self):
        """Test the __init__ method"""

        priors = (
            cimem.core.GaussianPrior([0, 0, 0], 0.1 * np.eye(3), np.eye(3)),
            cimem.core.GaussianPrior([1, 1, 1], 0.1 * np.eye(3), np.eye(3))
        )
        cluster = cimem.core.Cluster('test', [0, 1, 2], priors, 0)
        self.assertEqual(cluster.name, 'test')
        np.testing.assert_array_almost_equal(cluster.sources, [0, 1, 2])

        # The name must be a string.
        self.assertRaises(TypeError, cimem.core.Cluster,
                          1, [0, 1, 2], priors, 0)

        # The sources must be convertible to an array of ints and have 1D.
        self.assertRaises(TypeError, cimem.core.Cluster,
                          'test', 'abc', priors, 0)
        self.assertRaises(ValueError, cimem.core.Cluster,
                          'test', [[0, 1]], priors, 0)

        # The sample number must be a positive integer.
        self.assertRaises(TypeError, cimem.core.Cluster,
                          'test', [0, 1, 2], priors, 'a')
        self.assertRaises(ValueError, cimem.core.Cluster,
                          'test', [0, 1, 2], priors, -2)

        # The prior must be an iterable of GaussianPrior.
        self.assertRaises(TypeError, cimem.core.Cluster,
                          'test', [0, 1, 2], [0, 1, 2], 2)
        self.assertRaises(TypeError, cimem.core.Cluster,
                          'test', [0, 1, 2], priors[0], 2)


class TestGaussianPrior(unittest.TestCase):
    """Test the cimem.core.GaussianPrior class"""

    def test_init(self):
        """Test the __init__ method"""

        prior = cimem.core.GaussianPrior(np.zeros((3,)), np.ones((3, 3)),
                                         np.eye(3))
        self.assertTrue(isinstance(prior, cimem.core.GaussianPrior))

        # The mean and variance must be convertible to an array of floats.
        self.assertRaises(TypeError, cimem.core.GaussianPrior,
                          'abc', [[0, 0], [0, 0]], np.eye(2))
        self.assertRaises(TypeError, cimem.core.GaussianPrior,
                          [0, 0], [['a', 'b'], [0, 1]], np.eye(2))

        # The mean must be 1D and the variance 2D.
        self.assertRaises(ValueError, cimem.core.GaussianPrior,
                          [[0, 0], [0, 0]], [[0, 0], [0, 0]], np.eye(2))
        self.assertRaises(ValueError, cimem.core.GaussianPrior,
                          [0, 0], [0, 0], np.eye(2))

        # The mean and variance must match.
        self.assertRaises(ValueError, cimem.core.GaussianPrior,
                          [0, 0, 0], [[0, 0], [0, 0]], np.eye(2))

        # The forward operator must be convertible to an array of floats.
        self.assertRaises(TypeError, cimem.core.GaussianPrior,
                          [0, 0], np.eye(2), [['a', 'a'], ['b', 'b']])

        # The forward operator must be 2D
        self.assertRaises(ValueError, cimem.core.GaussianPrior,
                          [0, 0], np.eye(2), [0, 1])

        # The shape of the forward must match the mean.
        self.assertRaises(ValueError, cimem.core.GaussianPrior,
                          [0, 0], np.eye(2), np.eye(3))

    def test_evaluate(self):
        """Test the evaluate method"""

        prior = cimem.core.GaussianPrior(np.ones((2,)), np.eye(2), np.eye(2))
        value, gradient = prior.partition(np.array([1, 1]))
        self.assertAlmostEqual(value, np.exp(3))
        np.testing.assert_array_almost_equal(gradient, [2, 2])

    def test_data_partition(self):
        """Test the __mul__ method"""

        forward = np.full((1, 2), 2)
        prior = cimem.core.GaussianPrior(np.ones((2,)), np.eye(2), forward)
        value, gradient = prior.data_partition(np.array([1]))
        self.assertAlmostEqual(value, np.exp(8))
        np.testing.assert_array_almost_equal(gradient, [12])
