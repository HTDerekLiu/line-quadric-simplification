import numpy as np
import sys
import scipy
import scipy.sparse 
from . normalize_row import normalize_row
from . face_areas import face_areas
import numpy.matlib as matlib

def vertex_normals(V, F):
    """
    VERTEXNORMALS computes face area weighted vertex normal

    Input:
        V (|V|,3) numpy array of vertex positions
        F (|F|,3) numpy array of face indices
    Output:
        VN (|V|,3) numpy array of normalized vertex normal
    """
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = np.cross(vec1, vec2) / 2
    FN_normalized = normalize_row(FN+sys.float_info.epsilon)
    faceArea = face_areas(V,F)

    rowIdx = F.reshape(F.shape[0]*F.shape[1])
    colIdx = matlib.repmat(np.expand_dims(np.arange(F.shape[0]),axis=1),1,3).reshape(F.shape[0]*F.shape[1])
    weightData = matlib.repmat(np.expand_dims(faceArea,axis=1),1,3).reshape(F.shape[0]*F.shape[1])
    W = scipy.sparse.csr_matrix((weightData, (rowIdx, colIdx)), shape=(V.shape[0],F.shape[0]))
    VN = W*FN_normalized
    VN = normalize_row(VN)
    return VN