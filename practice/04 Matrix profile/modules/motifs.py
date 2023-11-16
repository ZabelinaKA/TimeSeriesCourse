import numpy as np
from modules.utils import *


def top_k_motifs(matrix_profile, top_k=3): 
    """ 
    Find the top-k motifs based on matrix profile. 
 
    Parameters 
    --------- 
    matrix_profile : dict 
        The matrix profile structure. 
 
    top_k : int 
        Number of motifs. 
 
    Returns 
    -------- 
    motifs : dict 
        Top-k motifs (left and right indices and distances). 
    """ 
 
    motifs_idx = [] 
    motifs_dist = [] 
 
    for index in range(0, top_k): 
      motif_idx = np.argsort(matrix_profile['mp'])[index] 
      nearest_neighbor_idx = matrix_profile['mpi'][motif_idx] 
      if (motif_idx > nearest_neighbor_idx): 
        motifs_idx.append([nearest_neighbor_idx, motif_idx]) 
      else: 
        motifs_idx.append([motif_idx, nearest_neighbor_idx]) 
      motifs_dist.append(matrix_profile['mp'][motif_idx]) 
  
    return { 
        "indices" : motifs_idx, 
        "distances" : motifs_dist 
        }
