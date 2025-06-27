import numpy as np
from .massmatrix import massmatrix

def vertex_areas(V,F):
    return massmatrix(V,F).diagonal()

