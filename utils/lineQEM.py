import numpy as np
import heapq

from .global_variables import INF_COST, INVALID_VERTEX_INDEX, INF

from .adjacency_list_vertex_edge import adjacency_list_vertex_edge
from .adjacency_list_edge_face import adjacency_list_edge_face
from .adjacency_list_vertex_face import adjacency_list_vertex_face
from .face_areas import face_areas
from .compute_triangle_planes import compute_triangle_planes
from .remove_unreferenced import remove_unreferenced
from .vertex_areas import vertex_areas
from .vertex_normals import vertex_normals
from .face_normals import face_normals
from .edge_normals import edge_normals
from .edge_areas import edge_areas
from .edges import edges

from .single_edge_collapse_topology_update import single_edge_collapse_topology_update

def lineQEM(
        Vo: np.array, # (V, 3) vertex locations
        Fo: np.array, # (F, 3) face indices
        num_target_faces: int,
        line_quadric_weight: float = 1e-3, # threshold to switch to quadric scaling
        boundary_quadric_weight: float = 1.0, # for boundary perservation
        quadric_scalings: np.array = np.array([]), # (V,) scaling for vertex error metrics
        ):

    V = Vo.copy()
    F = Fo.copy()
    E = edges(F)

    GHOST_VERTEX_LOCATION = V.mean(0)

    nV = V.shape[0]
    nF = F.shape[0]
    nE = E.shape[0]

    if len(quadric_scalings) == 0:
        quadric_scalings = np.zeros(V.shape[0]) # set it to one to have no influence

    # compute topological relationship (store everything for speed purposes)
    V2E = adjacency_list_vertex_edge(E)
    V2F = adjacency_list_vertex_face(F)
    E2F = adjacency_list_edge_face(E,F)

    # get quadrics
    Q = compute_vertex_quadrics(V,F,E,E2F,quadric_scalings,boundary_quadric_weight,line_quadric_weight)

    # decimation parameters
    total_collapses = nF - num_target_faces
    cur_collapses = 0

    # construct priority queue
    min_heap = [] 
    for e in range(nE):
        i,j = E[e,:]
        cost, v_opt = optimal_location_and_cost(Q, i, j)

        random_number_to_break_tie = np.random.rand()
        heapq.heappush(min_heap, (cost, random_number_to_break_tie, v_opt, e, i, j, cur_collapses, cur_collapses))

    # start decimation
    V_time_stamps = np.zeros(nV, dtype=np.int32) # keep track of the latest cost
    V_is_removed = np.zeros(nV, dtype=bool)
    while True:
        if cur_collapses >= total_collapses:
            break
        if len(min_heap) == 0:
            print("empty heap, cannot be decimated further")
            break

        cost, _, v_opt, e, i, j, time_stamp_i, time_stamp_j = heapq.heappop(min_heap)
        assert(i != j)

        # =========================
        # CHECK if this edge information is up-to-date 
        # =========================
        if V_is_removed[i] == True or V_is_removed[j] == True:
            continue
        if time_stamp_i != V_time_stamps[i]: # if cost is obsolete
            continue 
        if time_stamp_j != V_time_stamps[j]: # if cost is obsolete
            continue 
        if np.abs(cost-INF_COST) < 1e-6:
            print("encounter INF cost, cannot be decimated further")
            break
        assert(i == E[e,0]) # a debugging assertion to make sure (e,i,j) is up to date
        assert(j == E[e,1])

        # =========================
        # start post collapse
        # =========================
        nf_remove = np.intersect1d(V2F[i], V2F[j])
        cur_collapses += len(nf_remove)
        if cur_collapses % 1000 == 0:
            print("decimation progress:", cur_collapses, "/", total_collapses)

        single_edge_collapse_topology_update(i,j, E,F,V2E,V2F,E2F)

        # move vertex
        V[i,:] = v_opt
        V[j,:] = GHOST_VERTEX_LOCATION
        
        # mark removed vertices
        V_is_removed[j] = True

        # update vertex quadrics 
        Q[i] = Q[i] + Q[j] 
        Q[j] = INF

        # insert one-ring vertices to the queue
        one_ring_edges = V2E[i]
        V_time_stamps[i] = cur_collapses
        for e_ in one_ring_edges:
            if E[e_,0] == i:
                j_ = E[e_,1]
            elif E[e_,1] == i:
                j_ = E[e_,0]
            else:
                raise ValueError("A BUG: vertex i doesn't exist in edge e_")
            
            cost_, v_opt_ = optimal_location_and_cost(Q, i, j_)

            random_number_to_break_tie = np.random.rand()
            if i < j_:
                heapq.heappush(min_heap, (cost_, random_number_to_break_tie, v_opt_, e_, i, j_, cur_collapses, V_time_stamps[j_]))
            else:
                heapq.heappush(min_heap, (cost_, random_number_to_break_tie, v_opt_, e_, j_, i, V_time_stamps[j_], cur_collapses))


    f_valid = np.where(F[:,0] != INVALID_VERTEX_INDEX)[0]
    V,F,IMV,vIdx = remove_unreferenced(V,F[f_valid,:])
    return V,F

def compute_boundary_quadric(V,E,fn,e):
    # compute edge normal (the direction doens't matter as we compoute squared distance)
    vi, vj = V[E[e,0]], V[E[e,1],:]
    en = np.cross(vj - vi, fn)
    en = en / (np.linalg.norm(en) + 1e-12)

    # compute edge plane equation
    ed = -en.dot(vi)
    ep = np.append(en,ed) # edge plane

    return np.outer(ep, ep)

def compute_line_quadric(v, vn):
    # get one normal vector 
    if np.abs(vn[0]) < 0.9:
        n0 = np.array([1.,0,0])
    else:
        n0 = np.array([0,1.,0])
    n0 = n0 - (n0.dot(vn) / vn.dot(vn)) * vn
    n0 = n0 / np.linalg.norm(n0)

    # get another normal vector 
    n1 = np.cross(n0, vn)

    # compute edge plane equation
    ep0 = np.append(n0,-n0.dot(v)) # edge plane
    Q0 = np.outer(ep0, ep0)

    ep1 = np.append(n1,-n1.dot(v)) # edge plane
    Q1 = np.outer(ep1, ep1)
    return Q0 + Q1

def compute_vertex_quadrics(V,F,E,E2F,
                            quadric_scalings,
                            boundary_quadric_weight,
                            line_quadric_weight):
    # precompute some useful quantities
    FN = face_normals(V,F)
    VN = vertex_normals(V,F)
    EN = edge_normals(V,F,E)

    FA = face_areas(V,F) 
    VA = vertex_areas(V,F)
    EA = edge_areas(V,F,E)

    # face quadrics
    p = compute_triangle_planes(V, F)
    Qf = np.einsum("fi,fj->fij", p, p) # face quadrics (nF,4,4)

    # vertex quadrics
    Qv = np.zeros((V.shape[0],4,4))
    for f in range(F.shape[0]):
        for vIdx in F[f,:]:
            Qv[vIdx,:,:] += Qf[f,:,:] * FA[f] / 3.0

    # add boundary edge quadrics from "Simplifying Surfaces with Color and Texture using Quadric Error Metrics" by Garland & Heckbert 1998
    for e in range(E.shape[0]):
        if is_boundary_edge(e, E2F): # if this is boundary half edge
            # compute boundary quadric
            f = E2F[e][0]
            assert(F[f,0] != INVALID_VERTEX_INDEX)
            Be = compute_boundary_quadric(V,E,FN[f,:],e)

            ea = EA[e]

            Qv[E[e,0],:,:] += Be * ea * boundary_quadric_weight
            Qv[E[e,1],:,:] += Be * ea * boundary_quadric_weight
    
    # add uniform weight line quadrics
    for v in range(V.shape[0]):
        Ql = compute_line_quadric(V[v,:], VN[v,:])
        Qv[v,:,:] += Ql * VA[v] * line_quadric_weight

    # apply scaling for extreme weights (see Sec 4.4.1)
    Qv = Qv * quadric_scalings[:,None,None]

    return Qv

def is_boundary_edge(e, E2F):
    return (len(E2F[e]) == 1)

def optimal_location_and_cost(Q, i, j):
    # combined quadrics
    Qe = Q[i] + Q[j]
    A = Qe[:3,:3]
    b = Qe[:3,3]
    c = Qe[3,3]

    v_opt = np.linalg.solve(A @ A.T, A.T @ (-b))
    cost = (v_opt @ A).dot(v_opt) + 2 * b.dot(v_opt) + c
    return cost, v_opt
