import unittest

import numpy as np

import cimem
import cimem.core

from bayesnet import ProbabilityMassFunction


class TestSolve(unittest.TestCase):
    """Test the cimem.solve function"""

    def test_one_cluster_one_source(self):
        """Test using a single cluster with a single source"""

        nb_samples = 1
        source_intensities = np.full((nb_samples, 1), 1.5)

        # The model is a single cluster with one source.
        forward = np.eye(1)
        priors = (
            cimem.core.GaussianPrior([0], [[0.1]], forward),
            cimem.core.GaussianPrior([1], [[0.1]], forward)
        )
        clusters = [cimem.core.Cluster('A', [0], priors, 0)]

        # Solve the MEM problem and reconstruct the source intensity.
        lagrange, marginals = cimem.solve(source_intensities, clusters)
        intensities = cimem.reconstruct_source_intensities(
            marginals, clusters, 1, 1, nb_samples, lagrange)
        np.testing.assert_array_almost_equal(intensities, source_intensities)

    def test_one_cluster_two_sources(self):
        """Test using single cluster with two sources"""

        nb_sensors = 2
        nb_sources = 2
        nb_samples = 1
        source_intensities = np.full((nb_sources, nb_samples), 1.0)

        # The model is a single cluster with two sources.
        forward = np.eye(nb_sensors)
        priors = (
            cimem.core.GaussianPrior([0, 0], 0.1 * np.eye(2), forward),
            cimem.core.GaussianPrior([1, 1], 0.1 * np.eye(2), forward)
        )
        clusters = [cimem.core.Cluster('A', [0, 1], priors, 0)]

        # Solve the MEM problem and reconstruct the source intensities.
        lagrange, marginals = cimem.solve(source_intensities, clusters)
        intensities = cimem.reconstruct_source_intensities(
            marginals, clusters, nb_sensors, nb_sources, nb_samples, lagrange)
        np.testing.assert_array_almost_equal(intensities, source_intensities)

    def test_one_cluster_with_forward(self):
        """Test using a single cluster with two sources but one observation"""

        nb_sensors = 1
        nb_sources = 2
        nb_samples = 1
        source_intensities = np.full((nb_sources, nb_samples), 1.0)
        forward = np.array([[1.0, 1.0]])
        data = np.dot(forward, source_intensities)

        # The model is a single cluster with two sources.
        priors = (
            cimem.core.GaussianPrior([0, 0], 0.1 * np.eye(2), forward),
            cimem.core.GaussianPrior([1, 1], 0.1 * np.eye(2), forward)
        )
        clusters = [cimem.core.Cluster('A', [0, 1], priors, 0)]

        # Solve the MEM problem and reconstruct the source intensity.
        lagrange, marginals = cimem.solve(data, clusters)
        intensities = cimem.reconstruct_source_intensities(
            marginals, clusters, nb_sensors, nb_sources, nb_samples, lagrange)
        np.testing.assert_array_almost_equal(intensities, source_intensities)

    def test_two_independent_clusters(self):
        """Test using two independent clusters"""

        nb_sensors = 2
        nb_sources = 2
        nb_samples = 1
        source_intensities = np.array([[-1.0], [2.0]])
        forward = np.eye(nb_sources)
        data = np.dot(forward, source_intensities)

        # The model is a single cluster with two sources.
        priors_cluster_1 = (
            cimem.core.GaussianPrior([0], 0.1 * np.eye(1), forward[:, 0:1]),
            cimem.core.GaussianPrior([1], 0.1 * np.eye(1), forward[:, 0:1])
        )
        priors_cluster_2 = (
            cimem.core.GaussianPrior([0], 0.1 * np.eye(1), forward[:, 1:]),
            cimem.core.GaussianPrior([1], 0.1 * np.eye(1), forward[:, 1:])
        )
        clusters = [
            cimem.core.Cluster('A', [0], priors_cluster_1, 0),
            cimem.core.Cluster('B', [1], priors_cluster_2, 0)
        ]

        # Solve the MEM problem and reconstruct the source intensity.
        lagrange, marginals = cimem.solve(data, clusters)
        intensities = cimem.reconstruct_source_intensities(
            marginals, clusters, nb_sensors, nb_sources, nb_samples, lagrange)
        np.testing.assert_array_almost_equal(intensities, source_intensities)

    def test_with_two_time_samples(self):
        """Test using two time samples"""

        nb_sensors = 1
        nb_sources = 1
        nb_samples = 2
        source_intensities = np.array([[0.0, 1.0]])  # One source two times.
        forward = np.eye(nb_sources)
        data = np.dot(forward, source_intensities)

        # The model is two clusters, one for each sample.
        priors_cluster_1 = (
            cimem.core.GaussianPrior([0], 0.1 * np.eye(1), forward),
            cimem.core.GaussianPrior([1], 0.1 * np.eye(1), forward)
        )
        priors_cluster_2 = (
            cimem.core.GaussianPrior([0], 0.1 * np.eye(1), forward),
            cimem.core.GaussianPrior([1], 0.1 * np.eye(1), forward)
        )
        clusters = [
            cimem.core.Cluster('A', [0], priors_cluster_1, 0),
            cimem.core.Cluster('B', [0], priors_cluster_2, 1)
        ]

        # Solve the MEM problem and reconstruct the source intensity.
        lagrange, marginals = cimem.solve(data, clusters)
        intensities = cimem.reconstruct_source_intensities(
            marginals, clusters, nb_sensors, nb_sources, nb_samples, lagrange)
        np.testing.assert_array_almost_equal(intensities, source_intensities)

    def test_solve_vs_pinv(self):
        """Test the solve method by comparing it to the pseudoinverse"""

        nb_sensors = 32
        nb_sources = 32
        nb_samples = 1

        # Build a simple problem.
        forward = np.random.randn(nb_sensors, nb_sources)
        source_intensities = np.random.randn(nb_sources, nb_samples)
        source_intensities[nb_sources // 2:] = 0
        measurements = np.dot(forward, source_intensities)

        # A single Gaussian prior for all sources. This makes to solution
        # equal to the minimum norm solution.
        priors = (
            cimem.core.GaussianPrior(np.zeros((nb_sources,)),
                                     np.zeros((nb_sources, nb_sources)),
                                     forward),
            cimem.core.GaussianPrior(np.zeros((nb_sources,)),
                                     np.eye(nb_sources), forward)
        )
        clusters = [
            cimem.core.Cluster('all-sources', range(nb_sources), priors, 0)]

        lagrange, marginals = cimem.solve(measurements, clusters)
        recovered_cimem = cimem.reconstruct_source_intensities(
            marginals, clusters, nb_sensors, nb_sources, nb_samples, lagrange)

        # The pseudo inverse solution.
        pseudoinverse = np.linalg.pinv(forward)
        recovered_pinv = np.dot(pseudoinverse, measurements)

        # The solution should fit the measurements.
        np.testing.assert_array_almost_equal(
            measurements, np.dot(forward, recovered_cimem))

        # The solutions should be almost identical.
        np.testing.assert_array_almost_equal(
            recovered_cimem, recovered_pinv, 5)

    def test_add_pmfs(self):
        """Test using additional pmfs"""

        nb_sensors = 32
        nb_sources = 32
        nb_samples = 1

        # Build a simple problem.
        forward = np.random.randn(nb_sensors, nb_sources)
        source_intensities = np.random.randn(nb_sources, nb_samples)
        measurements = np.dot(forward, source_intensities)

        # A single Gaussian prior for all sources. This makes to solution
        # equal to the minimum norm solution.
        priors = (
            cimem.core.GaussianPrior(np.zeros((nb_sources,)),
                                     np.zeros((nb_sources, nb_sources)),
                                     forward),
            cimem.core.GaussianPrior(np.zeros((nb_sources,)),
                                     np.eye(nb_sources), forward)
        )
        clusters = [
            cimem.core.Cluster('all-sources', range(nb_sources), priors, 0)]

        # Add pmfs which forces activity.
        pmfs = [ProbabilityMassFunction(clusters, [0.0, 1.0])]

        lagrange, marginals = cimem.solve(measurements, clusters, pmfs)
        recovered_cimem = cimem.reconstruct_source_intensities(
            marginals, clusters, nb_sensors, nb_sources, nb_samples, lagrange)

        # The solution should fit the measurements.
        np.testing.assert_array_almost_equal(
            measurements, np.dot(forward, recovered_cimem), 3)

        # The cluster must be fully active.
        np.testing.assert_array_almost_equal(
            marginals[clusters[0]].probabilities, [0.0, 1.0])
