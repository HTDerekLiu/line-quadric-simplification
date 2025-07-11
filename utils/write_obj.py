import numpy as np

def write_obj(fileName,V,F,vertex_colors=None):
    """
    write .obj file

    Input:
    filepath a string of mesh file path
    V: |V|x3 array of vertex positions
    F: |F|x3 array of face indices

    Output:
    an .obj file
    """
    if V.shape[1] == 2: # if 2d mesh
        V = np.hstack((V, np.zeros((V.shape[0],1))))
    f = open(fileName, 'w')
    for ii in range(V.shape[0]):
        if vertex_colors is not None:
            assert(vertex_colors.shape == V.shape)
            string = 'v ' + str(V[ii,0]) + ' ' + str(V[ii,1]) + ' ' + str(V[ii,2]) + ' ' + str(vertex_colors[ii,0]) + ' ' + str(vertex_colors[ii,1]) + ' ' + str(vertex_colors[ii,2]) + '\n'
        else:
            string = 'v ' + str(V[ii,0]) + ' ' + str(V[ii,1]) + ' ' + str(V[ii,2]) + '\n'
        f.write(string)
    Ftemp = F + 1
    for ii in range(F.shape[0]):
        string = 'f ' + str(Ftemp[ii,0]) + ' ' + str(Ftemp[ii,1]) + ' ' + str(Ftemp[ii,2]) + '\n'
        f.write(string)
    f.close()

    