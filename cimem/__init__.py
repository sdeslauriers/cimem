import json
import logging
import logging.config
import os

from typing import Sequence, Tuple

import numpy as np

from scipy.optimize import minimize
from bayesnet import ProbabilityMassFunction
from bayesnet.junction import Marginals

from cimem.core import Cluster


# Load logging configuration.
config_file = os.path.join(os.path.dirname(__file__),
                           '..', 'config', 'logging.json')
logging.config.dictConfig(json.load(open(config_file, 'rt')))
logger = logging.getLogger(__name__)


def solve(data: np.ndarray, clusters: Sequence[Cluster],
          pmfs: Sequence[ProbabilityMassFunction] = None,
          covariance: np.ndarray = None) \
        -> Tuple[np.ndarray, Marginals]:
    """Solves the inverse problem using CIMEM

    Solves the inverse problem using Connectivity Informed Maximum Entropy
    on the Mean (CIMEM).

    Args:
        data: The M/EEG data in a 1D numpy array of floats.
        clusters: The source clusters used to spatially regularize the problem.
        pmfs: Other probability mass table to take into account when fitting
            the data. This is where additional priors are included into the
            problem.

    Returns:
        lagrange: The optimal Lagrange multipliers.
        marginals: The marginals of the Bayesian network used to define the
            priors.

    """

    if data.ndim == 1:
        data = data[:, None]

    if pmfs is None:
        pmfs = []

    if covariance is None:
        covariance = np.zeros((data.size, data.size))

    # Flatten the data.
    nb_sensors, nb_samples = data.shape
    flat_data = data.ravel('F')
    sensors = np.arange(nb_sensors)

    # For each cluster, add an evidence table.
    evidence_tables = [ProbabilityMassFunction([c]) for c in clusters]
    nb_states = np.sum([len(c) for c in clusters])

    # Compute the marginals.
    logger.info('Computing marginals.')
    marginals = Marginals(pmfs + evidence_tables)

    def cost(lagrange):

        # Update the evidence and marginals for the current Lagrange
        # multipliers.
        gradients = np.zeros((nb_states, nb_sensors))
        locations = np.zeros((nb_states, nb_sensors), dtype=int)
        i = 0
        for cluster, evidence_table in zip(clusters, evidence_tables):
            location = sensors + nb_sensors * cluster.sample
            values, gradient = zip(*(p.data_partition(lagrange[location])
                                     for p in cluster.priors))

            gradients[i:i + len(cluster), :] = gradient
            locations[i:i + len(cluster), :] = location
            i += len(cluster)

            normalization = np.sum(values)
            evidence_table.probabilities = values / normalization
            evidence_table.normalization = normalization

        marginals.update()

        # Compute the MEM cost.
        temp = np.dot(covariance, lagrange)
        entropy = 0.5 * np.dot(lagrange, temp) - np.dot(lagrange, flat_data)
        entropy += np.log(marginals.normalization)

        weights = [p for c in clusters for p in marginals[c].probabilities]
        weighted_gradients = np.array(weights)[:, None] * gradients
        gradient = temp - flat_data
        for location, cluster_gradient in zip(locations, weighted_gradients):
            gradient[location] += cluster_gradient

        return entropy, gradient

    logger.info('Optimizing cost function.')
    res = minimize(cost, np.zeros_like(flat_data),
                   jac=True,
                   options={'gtol': 1e-6},
                   method='CG')

    # Update the marginals at the optimal solution.
    cost(res.x)

    logger.info('Done.')
    return res.x, marginals


def reconstruct_source_intensities(
        marginals: Marginals, clusters: Sequence[Cluster],
        nb_sensors: int, nb_sources: int, nb_samples: int,
        lagrange: np.ndarray) -> np.ndarray:
    """Reconstructs the source intensities

    Reconstructs the source intensities for a given value of the Lagrange
    multipliers. If the Lagrange multipliers used are those returned by the
    solve function, this corresponds to the CIMEM source estimations.

    Args:
        marginals: The marginals for each variable of the Bayesian network.
        clusters: The clusters used to spatially regularize the problem.
        nb_sensors: The number of sensors of the data.
        nb_sources: The number of sources of the model.
        nb_samples: The number of samples of the data.
        lagrange: The Lagrange multipliers where the sources are
            reconstructed. Usually those returned by the solve function.

    Returns:
        intensities: The reconstructed source intensities.

    """

    # Initialize the source intensities.
    intensities = np.zeros((nb_sources, nb_samples))

    sensors = np.arange(nb_sensors)
    for cluster in clusters:

        # Compute the posterior of the cluster.
        posterior = marginals[cluster].probabilities

        # Compute the source intensities of the cluster.
        for i, prior in enumerate(cluster.priors):
            location = sensors + nb_sensors * cluster.sample
            prior_intensities = prior.derivative(lagrange[location])
            intensities[cluster.sources, cluster.sample] += \
                posterior[i] * prior_intensities

    return intensities
