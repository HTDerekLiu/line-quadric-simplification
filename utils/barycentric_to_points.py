import numpy as np

def barycentric_to_points(V,F,bc,bf):
    """
    givne a set of barycentric coordinates, compute the point locations

    Inputs
        V: V-by-3 array of vertex locations
        F: F-by-3 array of face list
        bc: P-by-3 array of barycentric coordinates
        bf: P list of face indices

    Outputs
        P: P-by-3 of barycentric point locations
    """
    if len(bf)==0:
        return np.array([],dtype=float)
    v0_index = F[bf,0]
    v1_index = F[bf,1]
    v2_index = F[bf,2]

    v0 = V[v0_index,:]
    v1 = V[v1_index,:]
    v2 = V[v2_index,:]

    return bc[:,[0]]*v0 + bc[:,[1]]*v1 + bc[:,[2]]*v2