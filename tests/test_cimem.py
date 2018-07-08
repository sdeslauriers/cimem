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
            marginals, clusters, nb_samples, lagrange)
        np.testing.assert_array_almost_equal(intensities, source_intensities)

    def test_one_cluster_two_sources(self):
        """Test using single cluster with two sources"""

        nb_samples = 2
        source_intensities = np.full((nb_samples,), 1.0)

        # The model is a single cluster with two sources.
        clusters = [cimem.core.Cluster('A', [0, 1], 0)]

        # Solve the MEM problem and reconstruct the source intensities.
        lagrange, marginals = cimem.solve(source_intensities, clusters)
        intensities = cimem.reconstruct_source_intensities(
            marginals, clusters, nb_samples, lagrange)
        np.testing.assert_array_almost_equal(intensities, source_intensities)

    def test_one_cluster_with_forward(self):
        """Test using a single cluster with two sources but one observation"""

        nb_sources = 2
        source_intensities = np.full((nb_sources,), 1.0)
        forward = np.array([[1.0, 1.0]])
        data = np.dot(forward, source_intensities)

        # The model is a single cluster with two sources.
        clusters = [cimem.core.Cluster('A', [0, 1], 0, forward)]

        # Solve the MEM problem and reconstruct the source intensity.
        lagrange, marginals = cimem.solve(data, clusters)
        intensities = cimem.reconstruct_source_intensities(
            marginals, clusters, nb_sources, lagrange)
        np.testing.assert_array_almost_equal(intensities, source_intensities)
