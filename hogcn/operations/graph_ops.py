import h5py
import numpy as np
import networkx as nx
import tensorflow as tf
from scipy import sparse

def get_graph(edges):
    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(graph.selfloop_edges())

    return graph

def get_normalized_adjacency_matrix(graph):
    A = nx.adjacency_matrix(graph)
    I = sparse.eye(A.shape[0])
    A_norm = A + I
    deg = A_norm.sum(axis=0).tolist()
    D = sparse.diags(deg, [0]).power(-0.5)

    return sparse.coo_matrix(D.dot(A_norm).dot(D))

def get_data(path):
    """
    TODO: Implement this using the Tensorflow dataset API.
    This will allow larger graphs to be read in without seeing
    an error about a tensor being to large.
    """
    with h5py.File(path, 'r') as hf:
        edges = hf['edges']
        feats = hf['nodes']
        labels = hf['labels']
        graph = get_graph(edges)
        feats = sparse.coo_matrix(feats)

        return graph, feats, np.array(labels)