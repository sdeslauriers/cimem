from typing import Sequence

import numpy as np

from bayesnet import DiscreteRandomVariable, ProbabilityMassFunction


class Cluster(DiscreteRandomVariable):
    def __init__(self, name: str, sources: Sequence[int], sample: int):
        """A source cluster

        An instance of the Cluster class represents a group of current
        sources in the brain.

        Args:
            name: The name of the cluster.
            sources: The source ids that are contained in the cluster.
            sample: The sample number of the cluster.

        Raises:
            TypeError if the name is not a String.
            TypeError if the sources cannot be converted to an array of
                integers.
            TypeError if the sample number cannot be converted to an integer.
            ValueError if the array of source ids has more that 1 dimension.

        """

        # Use the cluster name as its symbol and clusters are on or off.
        super().__init__(name, (0, 1))

        if not isinstance(name, str):
            raise TypeError('The name of the cluster must be a string, not {}.'
                            .format(type(name)))
        self._name = name

        try:
            sources = np.array(sources, dtype=np.int64)
        except (TypeError, ValueError):
            raise TypeError('The sources must be convertible to a numpy '
                            'array of integers.')

        if sources.ndim != 1:
            raise ValueError('The sources must have a shape of (N,), not {}.'
                             .format(sources.shape))
        self._sources = sources

        if not isinstance(sample, int):
            raise TypeError('The sample number of a cluster must be and '
                            'integer, not a {}.'.format(type(sample)))

        if sample < 0:
            raise ValueError('The sample number must be greater or equal'
                             'to 0.')
        self._sample = sample

        # Add default values for the priors.
        self._priors = (
            GaussianPrior(np.zeros((self.nb_sources,)),
                          0.1 * np.eye(self.nb_sources)),
            GaussianPrior(np.ones((self.nb_sources,)),
                          0.1 * np.eye(self.nb_sources)))

    @property
    def name(self) -> str:
        """Returns the name of the cluster"""
        return self._name

    @property
    def priors(self):
        """Returns the priors of the cluster"""
        return self._priors

    @property
    def sample(self):
        """Returns the sample number of the cluster"""
        return self._sample

    @property
    def nb_sources(self):
        """Returns the number of sources of the cluster"""
        return len(self._sources)

    @property
    def sources(self) -> np.ndarray:
        """Returns the source ids of the cluster"""
        return self._sources


class GaussianPrior(object):
    def __init__(self, mean: Sequence, variance: Sequence):
        """Gaussian prior on source intensities

        The cimem.GaussianPrior class represents the Gaussian priors on the
        intensities of a cluster of sources.

        Args:
            mean: The mean of each source of the cluster. Must be
                convertible to an array of floats with a shape of (N,).
            variance: The variance of the sources of the cluster. Must be
            convertible to an array of floats with a shape of (N, N).

        """

        try:
            mean = np.array(mean, np.float64)
        except (TypeError, ValueError):
            raise TypeError('The mean must be convertible to an array of '
                            'floats.')

        if mean.ndim != 1:
            raise ValueError('The mean must have a shape of (N,), not {}.'
                             .format(mean.shape))

        try:
            variance = np.array(variance, np.float64)
        except (TypeError, ValueError):
            raise TypeError('The variance must be convertible to an array of '
                            'floats.')

        if variance.ndim != 2 or variance.shape[0] != variance.shape[1]:
            raise ValueError('The mean must have a shape of (N, N), not {}.'
                             .format(variance.shape))

        # The shape of the mean and variance must match.
        if mean.shape[0] != variance.shape[0]:
            raise ValueError('The shape of the mean and variance must match '
                             '({} != {}).'
                             .format(mean.shape[0], variance.shape[0]))

        self._mean = mean
        self._variance = variance

    def __call__(self, lagrange: np.ndarray):
        """Evaluates the prior at the Lagrange multipliers

        Evaluates the prior at the Lagrange multipliers and returns both its
        value and its gradient divided by the value.

        For a Gaussian prior, this is:

            value = exp(L^t * m + 0.5 * L^t * v * L)

            gradient = (m + v * L) * exp(L^t * m + 0.5 * L^t * v * L) / value
                     = m + v * L

        where m in the mean, v is the variance and L is the Lagrange
        multiplier.

        Args:
            lagrange: The Lagrange multipliers where the prior is evaluated.

        """

        temp = np.dot(self._variance, lagrange)
        value = np.exp(np.dot(lagrange, self._mean) +
                       0.5 * np.dot(lagrange, temp))

        return value, self._mean + temp
