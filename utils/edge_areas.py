import numpy as np
from . face_areas import face_areas
from . adjacency_list_edge_face import adjacency_list_edge_face

def edge_areas(V, F, E):
    """
    this function computes face area weighted edge normal

    Input:
        V (|V|,3) numpy array of vertex positions
        F (|F|,3) numpy array of face indices
        E (|E|,2) numpy array of edge indices
    Output:
        EA (|E|,) numpy array of normalized edge areas
    """
    FA = face_areas(V,F)
    E2F = adjacency_list_edge_face(E,F)
    EA = np.zeros((E.shape[0],))
    for e, fs in enumerate(E2F):
        EA[e] = np.sum(FA[fs])
    return EA