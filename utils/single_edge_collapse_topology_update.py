import numpy as np
from .global_variables import INVALID_VERTEX_INDEX

def single_edge_collapse_topology_update(i, j, E,F,V2E,V2F,E2F):
    """
    in-place update the topology of single edge collapse
    """
    # i,j = E[e,:]

    nf = np.union1d(V2F[i], V2F[j])
    nf_remove = np.intersect1d(V2F[i], V2F[j])
    nf_keep = np.setdiff1d(nf, nf_remove, assume_unique=True).astype(int)

    # change F
    nfj = V2F[j]
    F_nfj = F[nfj,:] # local face list
    F_nfj[F_nfj == j] = i # change j to i 
    F[nfj,:] = F_nfj # put them back in the face list

    # remove faces shared between nfi and nfj
    F[nf_remove,:] = INVALID_VERTEX_INDEX

    # change E2F & E
    ne = np.union1d(V2E[i], V2E[j]).astype(int)
    E_ne = E[ne,:] # local edge list
    E_ne[E_ne == j] = i # change j to i
    idx = E_ne[:,0] > E_ne[:,1] # change it so that E[:,0] < E[:,1]
    tmp = E_ne[idx, 0]
    E_ne[idx,0] = E_ne[idx,1] 
    E_ne[idx,1] = tmp
    E[ne,:] = E_ne # put them back in the edge list

    # handle non-manifold cases
    ne_keep_tuples = []
    ne_keep = []
    ne_remove = []
    for e in ne:
        eij_tuple = canonical_form(E[e,:])

        if eij_tuple[0] == eij_tuple[1]: # this edge gets collapsed to a point
            E2F[e] = [] 
            ne_remove.append(e)
            E[e,:] = INVALID_VERTEX_INDEX

        elif eij_tuple in ne_keep_tuples: # has duplicate edge in ne
            index_to_ne_keep = ne_keep_tuples.index(eij_tuple)
            e_already_there = ne_keep[index_to_ne_keep]
            assert(e_already_there != e)

            f_added_from_e = np.setdiff1d(E2F[e], nf_remove).astype(int)
            E2F[e_already_there] = np.union1d(E2F[e_already_there],f_added_from_e)
            
            E2F[e] = []
            ne_remove.append(e)
            E[e,:] = INVALID_VERTEX_INDEX
            
        else: # does not have duplicate edge in 
            E2F[e] = np.setdiff1d(E2F[e], nf_remove).astype(int)
            ne_keep_tuples.append(eij_tuple)
            ne_keep.append(e)

    # change V2E, V2F
    V2E[i] = np.array(ne_keep).astype(int)
    V2E[j] = []

    V2F[i] = nf_keep
    V2F[j] = []

    # insert one-ring vertices to the queue
    one_ring_vertices = np.unique(E[V2E[i],:])
    one_ring_vertices = np.setdiff1d(one_ring_vertices, i)
    for j_ in one_ring_vertices:
        V2E[j_] = np.setdiff1d(V2E[j_], ne_remove)
        V2F[j_] = np.setdiff1d(V2F[j_], nf_remove)

    return 


def canonical_form(input):
    """
    outputs the canonical form for each type of simplex.
    """
    if len(input) == 1: # input is a vertex
        return input
    elif len(input) == 2: # input edge
        i,j = input
        return (min(i,j), max(i,j))
    elif len(input) == 3: # input face
        ijk = np.array(input)
        shift = np.argmin(ijk)
        i,j,k = np.roll(ijk, -shift)
        out = (i,j,k)
        assert(out[0] < out[1])
        assert(out[0] < out[2])
        return out