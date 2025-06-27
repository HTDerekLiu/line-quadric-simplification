import numpy as np

def edges_with_mapping(F):
  '''
  EDGES compute edges from face information

  Input:
    F (|F|,3) numpy array of face indices
  Output:
    uE (|E|,2) numpy array of edge indices
    FS2E (|F|,3) numpy array mapping from halfedges to unique edges. 
    fs2E_sign numpy array indicating whether a fs is aligned with E or not:
    >> if fs2E_sign[f,s] == -1:
    >>   then the revserse of E[fs2E[f,s],:] is the tip vertices of the face side
    
  Note:
    Our halfedge convention identifies each halfedge by the index of the face and index to the starting vertex of the face. For instance, if 
      F[f,:] = [i, j, k]
    then
      F[f,0] = face side for directed edge (i,j)
  '''
  F12 = F[:, np.array([1,2])]
  F20 = F[:, np.array([2,0])]
  F01 = F[:, np.array([0,1])]
  hE = np.concatenate( (F12, F20, F01), axis = 0)

  hE_sorted = np.sort(hE, axis = 1)
  hE_sign_vec = ((hE[:,0] < hE[:,1]).astype(int) * 2) - 1
  hE_sign = hE_sign_vec.reshape(F.shape[1], F.shape[0]).T
  
  E, fs2E_vec = np.unique(hE_sorted, return_inverse=True, axis=0)
  fs2E = fs2E_vec.reshape(F.shape[1], F.shape[0]).T

  # adjust the index to make sure it is aligned with our fs convention (see above Note)
  fs2E = fs2E[:,[2,0,1]]
  fs2E_sign = hE_sign[:,[2,0,1]]

  return E, fs2E, fs2E_sign