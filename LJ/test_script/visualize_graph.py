import dgl.data
import matplotlib.pyplot as plt
import networkx as nx

import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl

dataset_path = "./graphs_to_train"

idx = 2033

fname = f'graph{idx}.dgl'
data_path = os.path.join(dataset_path, fname)
        
g = dgl.load_graphs(data_path)

print(g[0][0].ndata["e"])

options = {
    'node_color': 'red',
    'node_size': 30,
    'width': 0.1,
}

G = dgl.to_networkx(g[0][0])
plt.figure(figsize=[15,7])
nx.draw(G, **options)
plt.savefig('particle_graph.png')