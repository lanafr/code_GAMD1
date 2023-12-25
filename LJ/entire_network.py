import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl

import sys, os
sys.path.append(os.path.join('../',os.path.dirname(os.path.abspath(''))))
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
sys.path.append('/home/guests/lana_frkin/GAMDplus/code')
sys.path.append('/home/guests/lana_frkin/GAMDplus/code/LJ')
print(sys.path)

from nn_module import SimpleMDNetNew_GAMD, SimpleMDNetNew
from actual_gamd import ParticleNetLightning_GAMD, NUM_OF_ATOMS
from train_endtoend_autoencoder_nice import ParticleNetLightning
import torch
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn
from the_sequential_network import Learner

from einops import rearrange


def network_trajectory(start_pos,t):
    PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_prvipravi/checkpoint_29.ckpt'
    SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_prvipravi/scaler_29.npz'
    args = SimpleNamespace(use_layer_norm=False,
                        encoding_size=256,
                        hidden_dim=128,
                        edge_embedding_dim=1,
                        drop_edge=False,
                        conv_layer=4,
                        rotate_aug=False,
                        update_edge=False,
                        use_part=False,
                        data_dir='',
                        loss='mse')
    model = ParticleNetLightning(args).load_from_checkpoint(PATH, args=args)
    model.load_training_stats(SCALER_CKPT)
    model.cuda()
    model.eval()

    PATH2 = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/sequential_network_withprvipravi/checkpoint_40.ckpt'
    SCALER_CKPT2 = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/sequential_network_withprvipravi/scaler_40.npz'
    args2 = SimpleNamespace(use_layer_norm=False,
                        encoding_size=32,
                        hidden_dim=128,
                        drop_edge=False,
                        conv_layer=4,
                        rotate_aug=False,
                        update_edge=False,
                        use_part=False,
                        data_dir='',
                        loss='mse')
    model2 = Learner(args2).load_from_checkpoint(PATH2, args=args2)
    #model2.load_training_stats(SCALER_CKPT2)
    model2.cuda()
    model2.eval()


    ## which one to start with

    pos_hopefully_same, graph_em1, graph_em2, embed = model.predict_nextpos(start_pos)

    graph = model.make_a_graph(start_pos)

    next_embeddings = model2.ode_embed_func(embed,t-1)

    trajectory = model.decode_the_sequence(next_embeddings, graph, t-1)

    trajectory = [start_pos] + trajectory

    return trajectory

def rdf_func(coords, box_size, dr=0.1, r_max=None):
    num_particles = len(coords)
    print("Dimenzija je:")
    print(len(coords))
    if r_max is None:
        r_max = np.min(box_size) / 2.0  # Use half the minimum box size as a default

    coords = np.concatenate(coords)

    radii = np.arange(0, r_max + dr, dr)
    hist, _ = np.histogram(np.linalg.norm(coords - coords[:, np.newaxis], axis=-1), bins=radii)
    
    # Initialize g_r with zeros
    g_r = np.zeros_like(hist, dtype=float)

    # Correct for periodic boundary conditions
    tree = cKDTree(coords, boxsize=box_size)
    for i in range(num_particles):
        # Find neighbors considering PBC
        neighbors = tree.query_ball_point(coords[i], r_max, p=np.inf)
        
        for j in neighbors:
            if i != j:
                delta_r = coords[j] - coords[i]
                delta_r = np.where(np.abs(delta_r) > 0.5 * box_size, box_size - np.abs(delta_r), delta_r)
                distance = np.linalg.norm(delta_r)
                bin_index = np.digitize([distance], radii)[0] - 1  # Adjust for 0-based indexing
                if 0 <= bin_index < len(g_r):
                    g_r[bin_index] += 1

    # Normalize by the number of particles and the volume of each bin
    volume = 4/3 * np.pi * (radii[1:]**3 - radii[:-1]**3)
    g_r /= num_particles * volume

    return g_r, radii[:-1]  # Exclude the last bin edge for plotting

t_start = 500
t = 100

start_all = np.load(f'md_dataset/lj_data_to_test/data_0_{t_start}.npz')
start_pos = start_all['pos']

trajectory_real = []
trajectory_model = []

trajectory_model = network_trajectory(start_pos, t)
for i in range(t_start,t+t_start):
    everything = np.load(f'md_dataset/lj_data_to_test/data_0_{t_start}.npz')
    just_pos = everything['pos']
    trajectory_real.append(just_pos)


trajectory_model[1:] = [tensor.detach().cpu().numpy() for tensor in trajectory_model[1:]]

trajectory_real = np.concatenate(trajectory_real)
trajectory_model = np.concatenate(trajectory_model)

print("Trjectory model")
print(type(trajectory_model))

print("Trjectory real")
print(type(trajectory_real))

loss = np.mean((trajectory_model - trajectory_real)**2, axis=(0, 1))
print("Loss is:")
print(loss)

print("Difference is:")
print(trajectory_model-trajectory_real)

num_particles = 258
BOX_SIZE = 27.27
box_size = np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])

# Calculate RDF with periodic boundary conditions
g_r_real, radii_real = rdf_func(trajectory_real, box_size, dr=0.1)

# Plot RDF
plt.plot(radii_real, g_r_real, label='Real Data')
plt.xlabel('Distance')
plt.ylabel('Radial Distribution Function (RDF)')
plt.legend()
plt.savefig('rdf_graph_real.png')

# Calculate RDF with periodic boundary conditions
g_r_model, radii_model = rdf_func(trajectory_model, box_size, dr=0.1)

# Plot RDF
plt.plot(radii_model, g_r_model, label='Model Data')
plt.xlabel('Distance')
plt.ylabel('Radial Distribution Function (RDF)')
plt.legend()
plt.savefig('rdf_graph_model.png')

print("Finished")