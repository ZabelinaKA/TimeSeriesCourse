import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile, top_k=3):
    """
    Find the top-k discords based on matrix profile.

    Parameters
    ---------
    matrix_profile : dict
        The matrix profile structure.

    top_k : int
        Number of discords.

    Returns
    --------
    discords : dict
        Top-k discords (indices, distances to its nearest neighbor 
        and the nearest neighbors indices).
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    for index in range(0, top_k):
        discord_idx = np.argmax(matrix_profile['mp'])
        nearest_neighbor_idx = matrix_profile['mpi'][discord_idx]
        discords_idx.append(discord_idx)
        discords_nn_idx.append(nearest_neighbor_idx)
        discords_dist.append(matrix_profile['mp'][discord_idx])
        matrix_profile['mp'][discord_idx] = -np.inf

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
