import numpy as np
import sys
from . normalize_row import normalize_row
from . face_areas import face_areas
from . adjacency_list_edge_face import adjacency_list_edge_face

def edge_normals(V, F, E):
    """
    this function computes face area weighted edge normal

    Input:
        V (|V|,3) numpy array of vertex positions
        F (|F|,3) numpy array of face indices
        E (|E|,2) numpy array of edge indices
    Output:
        EN (|E|,3) numpy array of normalized edge normal
    """
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = np.cross(vec1, vec2) / 2
    FN = normalize_row(FN+sys.float_info.epsilon)
    FA = face_areas(V,F)

    E2F = adjacency_list_edge_face(E,F)
    EN = np.zeros((E.shape[0], 3))
    for e, fs in enumerate(E2F):
        en = FN[fs,:] * FA[fs][:,None]
        en = np.sum(en, 0)
        en = en / np.linalg.norm(en)
        EN[e,:] = en
    return EN