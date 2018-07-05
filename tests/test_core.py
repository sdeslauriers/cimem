import unittest

import numpy as np

import cimem.core


class TestCluster(unittest.TestCase):
    """Test the cimem.core.Cluster class"""

    def test_init(self):
        """Test the __init__ method"""

        cluster = cimem.core.Cluster('test', [1, 2, 3], 0)
        self.assertEqual(cluster.name, 'test')
        np.testing.assert_array_almost_equal(cluster.sources, [1, 2, 3])

        # The name must be a string.
        self.assertRaises(TypeError, cimem.core.Cluster, 1, [1, 2, 3], 0)

        # The sources must be convertible to an array of ints and have 1D.
        self.assertRaises(TypeError, cimem.core.Cluster, 'test', 'abc', 0)
        self.assertRaises(ValueError, cimem.core.Cluster, 'test', [[1, 2]], 0)

        # The sample number must be a positive integer.
        self.assertRaises(TypeError, cimem.core.Cluster,
                          'test', [1, 2, 3], 'a')
        self.assertRaises(ValueError, cimem.core.Cluster,
                          'test', [1, 2, 3], -2)


