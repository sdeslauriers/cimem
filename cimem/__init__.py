from typing import Sequence, Tuple

import numpy as np

from scipy.optimize import minimize
from bayesnet import ProbabilityMassFunction
from bayesnet.junction import Marginals

from cimem.core import Cluster


def solve(data: np.ndarray, clusters: Sequence[Cluster]) \
        -> Tuple[np.ndarray, Marginals]:
    """Solves the inverse problem using CIMEM

    Solves the inverse problem using Connectivity Informed Maximum Entropy
    on the Mean (CIMEM).

    Args:
        data: The M/EEG data in a 1D numpy array of floats.
        clusters: The source clusters used to spatially regularize the problem.

    Returns:
        lagrange: The optimal Lagrange multipliers.
        marginals: The marginals of the Bayesian network used to define the
            priors.

    """

    # For each cluster, add an evidence table.
    evidence_tables = [ProbabilityMassFunction([c]) for c in clusters]
    nb_states = np.sum([len(c) for c in clusters])

    # Compute the marginals.
    marginals = Marginals(evidence_tables)

    def cost(lagrange):

        # Update the evidence and marginals for the current Lagrange
        # multipliers.
        gradients = np.zeros((nb_states, len(lagrange)))
        i = 0
        for cluster, evidence_table in zip(clusters, evidence_tables):
            values, gradient = zip(*(p.data_partition(lagrange)
                                     for p in cluster.priors))
            normalization = np.sum(values)
            gradients[i:i + len(cluster), :] = gradient
            i += len(cluster)

            evidence_table.probabilities = values / normalization
            evidence_table.normalization = normalization

        marginals.update()

        # Compute the MEM cost.
        entropy = -np.dot(lagrange, data)
        entropy += np.log(marginals.normalization)

        weights = [p for c in clusters for p in marginals[c].probabilities]
        gradient = -data
        gradient += np.sum(np.array(weights)[:, None] * gradients, 0)

        return entropy, gradient

    res = minimize(cost, np.zeros_like(data),
                   jac=True,
                   options={'gtol': 1e-6, 'maxiter': 200},
                   method='CG')

    # Update the marginals at the optimal solution.
    cost(res.x)

    return res.x, marginals


def reconstruct_source_intensities(
        marginals: Marginals, clusters: Sequence[Cluster], nb_samples: int,
        lagrange: np.ndarray) -> np.ndarray:
    """Reconstructs the source intensities

    Reconstructs the source intensities for a given value of the Lagrange
    multipliers. If the Lagrange multipliers used are those returned by the
    solve function, this corresponds to the CIMEM source estimations.

    Args:
        marginals: The marginals for each variable of the Bayesian network.
        clusters: The clusters used to spatially regularize the problem.
        nb_samples: The number of samples of the data.
        lagrange: The Lagrange multipliers where the sources are
            reconstructed. Usually those returned by the solve function.

    Returns:
        intensities: The reconstructed source intensities.

    """

    # Initialize the source intensities.
    intensities = np.zeros((nb_samples,))

    for cluster in clusters:

        # Compute the posterior of the cluster.
        posterior = marginals[cluster].probabilities

        # Compute the source intensities of the cluster.
        for i, prior in enumerate(cluster.priors):
            prior_intensities = prior.derivative(lagrange)
            intensities[cluster.sources] += posterior[i] * prior_intensities

    return intensities
