import polyscope as ps
import numpy as np

from utils.read_obj import read_obj
from utils.lineQEM import lineQEM
from utils.write_obj import write_obj

V, F = read_obj("./meshes/roblox_logo.obj")
num_target_faces = int(F.shape[0] * 0.1)

VV,FF = lineQEM(V,F, num_target_faces,
                line_quadric_weight=1e-3,
                boundary_quadric_weight=1.0,
                quadric_scalings=np.ones(V.shape[0]))
# write_obj("output.obj", VV, FF)

ps.init()
ps.register_surface_mesh("Original Mesh", V, F, enabled=False)
ps.register_surface_mesh("Simplified Mesh", VV, FF, edge_width=1)
ps.show()