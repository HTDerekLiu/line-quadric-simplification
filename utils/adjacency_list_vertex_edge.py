import numpy as np

def adjacency_list_vertex_edge(E):
    """
    build a vertex-edge adjacency list such that V2E[vertex_index] = [adjacent_edge_indices]
    Inputs
    E: |E|x3 array of edge
    Outputs
    V2E: list of lists with so that V2F[v] = [ei, ej, ...]
    """
    nV = E.max()+1
    V2E = [[] for _ in range(nV)]
    for e in range(E.shape[0]):
        i, j = E[e,:]
        V2E[i].append(e)
        V2E[j].append(e)
    return V2E