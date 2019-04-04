import networkx as nx
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

colors = [
    'red',
    'blue',
    'green',
    'orange',
    'purple',
    'yellow',
    'pink'
]


with h5py.File('tData.h5', 'r') as hf:
    edges = hf['edges']
    labels = hf['labels']
    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(graph.selfloop_edges())
    color_map = []
    for n in graph.nodes:
        color_map.append(colors[np.argmax(labels[int(n)])])

    options = {
        'node_size': 5,
        'width': 1,
    }

    nx.draw(graph, node_color=color_map, **options)
    plt.show()
