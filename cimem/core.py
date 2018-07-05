from typing import Sequence

import numpy as np

from bayesnet import DiscreteRandomVariable


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

    @property
    def name(self) -> str:
        """Returns the name of the cluster"""
        return self._name

    @property
    def sample(self):
        """Returns the sample number of the cluster"""
        return self._sample

    @property
    def sources(self) -> np.ndarray:
        """Returns the source ids of the cluster"""
        return self._sources
