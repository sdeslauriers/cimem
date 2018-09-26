import unittest

import numpy as np

from bayesnet import ProbabilityMassFunction as pmf
from cimem import reconstruct_source_intensities
from cimem import solve
from cimem.core import Cluster
from cimem.core import GaussianPrior


class TestTemporalRegularizationProblem(unittest.TestCase):
    """Test the solver using a temporally regularized cluster problem"""

    def test_wide_forward(self):
        """Test using a wide forward operator"""

        nb_sensors = 12
        nb_sources = 64
        nb_samples = 2

        # Generate the measurement using an identity matrix
        forward = np.random.randn(nb_sensors, nb_sources)
        source_intensities = 0.05 * np.random.randn(nb_sources, nb_samples)
        source_intensities[nb_sources // 2:, :] += 1
        measurements = np.dot(forward, source_intensities)

        # Two independent clusters.
        ones = np.ones((nb_sources // 2,))
        eye = np.eye(nb_sources // 2)
        cluster_1_forward = forward[:, :nb_sources // 2]
        priors_1 = (
            GaussianPrior(-ones, eye, cluster_1_forward),
            GaussianPrior(ones, eye, cluster_1_forward)
        )
        cluster_2_forward = forward[:, nb_sources // 2:]
        priors_2 = (
            GaussianPrior(-ones, eye, cluster_2_forward),
            GaussianPrior(ones, eye, cluster_2_forward)
        )
        sources = np.arange(nb_sources // 2)
        clusters = [Cluster('c1', sources, priors_1, s)
                    for s in range(nb_samples)]
        sources = np.arange(nb_sources // 2, nb_sources)
        clusters += [Cluster('c2', sources, priors_2, s)
                     for s in range(nb_samples)]

        # The clusters are in the same state at every sample.
        pmfs = []
        pmfs += [pmf(clusters[i:i + 2], [0.49, 0.01, 0.01, 0.49])
                 for i in range(nb_samples - 1)]
        pmfs += [pmf(clusters[i:i + 2], [0.49, 0.01, 0.01, 0.49])
                 for i in range(nb_samples, 2 * nb_samples - 1)]
        pmfs += [pmf([clusters[i], clusters[i + nb_samples]],
                     [0.01, 0.49, 0.49, 0.01])
                 for i in range(nb_samples)]

        lagrange, marginals = solve(measurements, clusters, pmfs)
        recovered_intensities = reconstruct_source_intensities(
            marginals, clusters, nb_sensors, nb_sources, nb_samples, lagrange)

        # The normalization factor must be the same for all marginals.
        for cluster in clusters:
            self.assertAlmostEqual(
                marginals[cluster].normalization,
                marginals[clusters[0]].normalization, 5)

        # The measurements should be explained almost exactly.
        np.testing.assert_array_almost_equal(
            measurements, np.dot(forward, recovered_intensities))