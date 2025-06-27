import numpy as np
from .edges_with_mapping import edges_with_mapping

def edges(F):
	'''
	EDGES compute edges from face information

	Input:
	  F (|F|,3) numpy array of face indices
	Output:
	  E (|E|,2) numpy array of edge indices
	'''
	edge = edges_with_mapping(F)[0]
	return edge