import unittest

import numpy as np

import cimem
import cimem.core


class TestSolve(unittest.TestCase):
    """Test the cimem.solve function"""

    def test_one_cluster_one_source(self):
        """Test using a single cluster with a single source"""

        nb_samples = 1
        source_intensities = np.full((nb_samples,), 1.5)

        # The model is a single cluster with one source.
        clusters = [cimem.core.Cluster('A', [0], 0)]

        # Solve the MEM problem and reconstruct the source intensity.
        lagrange, marginals = cimem.solve(source_intensities, clusters)
        intensities = cimem.reconstruct_source_intensities(
            marginals, clusters, 1, lagrange)
        np.testing.assert_array_almost_equal(intensities, source_intensities)
