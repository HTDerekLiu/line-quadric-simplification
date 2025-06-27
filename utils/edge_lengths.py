import numpy as np

def edge_lengths(V, F):
    """
    Build edge lengths for each face-side

    Inputs
    V: |V|x3 array of vertex locations
    F: |F|x3 array of face indices 

    Outputs
    l: |F|x3 array of face-side lengths
    """
    l01 = np.sqrt(np.sum((V[F[:,1],:] - V[F[:,0],:])**2,1))
    l12 = np.sqrt(np.sum((V[F[:,2],:] - V[F[:,1],:])**2,1))
    l20 = np.sqrt(np.sum((V[F[:,0],:] - V[F[:,2],:])**2,1))
    return np.stack((l01, l12, l20)).T