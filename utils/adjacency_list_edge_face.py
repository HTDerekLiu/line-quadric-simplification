import numpy as np
from .adjacency_list_vertex_face import adjacency_list_vertex_face

def adjacency_list_edge_face(E,F):
    """
    build a edge face adjacency list such that E2F[edge index] = [adjacent face index]

    Inputs
        E: |E|x2 array of edge
        F: |F|x3 array of faces
    Outputs
        E2F: list of lists with 
    """
    V2F = adjacency_list_vertex_face(F)
    E2F = [[] for _ in range(E.shape[0])]
    for e in range(E.shape[0]):
        i,j = E[e,:]
        nfi = V2F[i]
        nfj = V2F[j]
        nf = np.unique( np.concatenate((nfi, nfj)) ).astype(int) # all neighbor faces
        for f in nf:
            num_overlapped_indices = np.sum(np.in1d(F[f,:], E[e,:]))
            if num_overlapped_indices == 2: # edge e is in face f
                E2F[e].append(f)
    return E2F