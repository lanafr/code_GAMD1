import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dgl
import matplotlib.pyplot as plt


import sys, os
sys.path.append(os.path.join('../',os.path.dirname(os.path.abspath(''))))
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
sys.path.append('/home/guests/lana_frkin/GAMDplus/code')
sys.path.append('/home/guests/lana_frkin/GAMDplus/code/LJ')
print(sys.path)

from nn_module import SimpleMDNetNew_GAMD, SimpleMDNetNew
from train_endtoend_autoencoder_nice import ParticleNetLightning
import torch
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn
#from RNN_sequential_network import Learner
from the_sequential_network1 import Learner

from einops import rearrange
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error

from mpl_toolkits.mplot3d import Axes3D

def network_trajectory(start_pos, start_vel,t):
    PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_velocities_works?/checkpoint_29.ckpt'
    SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/autoencoder_velocities_works?/scaler_29.npz'
    args = SimpleNamespace(use_layer_norm=False,
                        encoding_size=32,
                        hidden_dim=128,
                        edge_embedding_dim=32,
                        drop_edge=True,
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

    PATH2 = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/sequential_network_original/checkpoint_39.ckpt'
    SCALER_CKPT2 = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/sequential_network_original/scaler_39.npz'
    args2 = SimpleNamespace(encoding_size=32)
    model2 = Learner(args2).load_from_checkpoint(PATH2, args=args2)
    #model2.load_training_stats(SCALER_CKPT2)
    model2.cuda()
    model2.eval()


    ## which one to start with

    with torch.no_grad():

        pos_hopefully_same, embed, vel = model.embed_pos(start_pos, start_vel)

        # graph = model.make_a_graph(start_pos)
            
        next_embeddings = model2.ode_embed_func(embed,t-1)

        # see how well it preformes (test)

        trajectory = model.decode_the_sequence(next_embeddings, t-1)

        trajectory = [start_pos] + trajectory

        """

        for i in range(0,t-1):

            data = np.load(f'md_dataset/lj_data/data_9_{i+500}.npz')
            pos_data = data['pos']
            vel_data = data['vel']
            #pos_hopefully_same, graph1, graph2, embed, force = model.predict_nextpos(pos_data)
            pred, emb, vel = model.embed_pos(pos_data, vel_data)
            embeddings_real.append(emb.detach().cpu().numpy())

        embeddings_model_np= np.stack(next_embeddings.detach().cpu().numpy()[0:t-1], axis=0)
        embeddings_real_np = np.stack(embeddings_real[0:t-1], axis=0)

        #print("Test:")
        #print(embeddings_model_np[33]-embeddings_real_np[33])

        embeddings_real_tensor = torch.tensor(embeddings_real_np)
        embeddings_model_tensor = torch.tensor(embeddings_model_np)

        diff = nn.MSELoss()(embeddings_real_tensor,embeddings_model_tensor)

        print("Diff in embeddings is:")
        print(diff)
        """

    return trajectory

def rdf_func(coords, box_size, dr=0.1, r_max=None):
    num_particles = len(coords)
    if r_max is None:
        r_max = np.min(box_size) / 2.0  # Use half the minimum box size as a default

    radii = np.arange(0, r_max + dr, dr)
    hist, _ = np.histogram(np.linalg.norm(coords - coords[:, np.newaxis], axis=-1), bins=radii)
    
    # Initialize g_r with zeros
    g_r = np.zeros_like(hist, dtype=float)

    print("Hejhej")

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

num_particles = 258
BOX_SIZE = 27.27
box_size = np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])

t_start = 500
t = 99

start_all = np.load(f'md_dataset/lj_data/data_5_{t_start}.npz')
start_pos = start_all['pos']
start_vel = start_all['vel']

trajectory_real = []
trajectory_model = []

for i in range(t_start,t+t_start):
    everything = np.load(f'md_dataset/lj_data/data_5_{i}.npz')
    just_pos = everything['pos']
    trajectory_real.append(just_pos)

trajectory_model = network_trajectory(start_pos, start_vel, t)

print("omggg")


#trajectory_model[1:] = [tensor.detach().cpu().numpy() for tensor in trajectory_model[1:]]

trajectory_real_np= np.stack(trajectory_real, axis=0)
trajectory_model_np = np.stack(trajectory_model, axis=0)

trajectory_model_np = np.mod(trajectory_model_np,BOX_SIZE)

trajectory_real_tensor = torch.tensor(trajectory_real_np)
trajectory_model_tensor = torch.tensor(trajectory_model_np)

loss = nn.MSELoss()(trajectory_real_tensor,trajectory_model_tensor)

print("Loss is:")
print(loss)

print("Difference is:")
print(trajectory_model_np-trajectory_real_np)

t_for_rdf = 49

## for one time snapshot

if (t_for_rdf>t): print("Time for rdf has to be smaller than t")

# Calculate RDF with periodic boundary conditions
g_r_real, radii_real = rdf_func(trajectory_real_np[t_for_rdf], box_size, dr=0.1)


# Plot RDF
plt.plot(radii_real, g_r_real, label='Real Data')
plt.xlabel('Distance')
plt.ylabel('Radial Distribution Function (RDF)')
plt.legend()
#plt.savefig('rdf_graph_real.png')

trajectory_model_np = np.mod(trajectory_model_np,BOX_SIZE)

g_r_model, radii_model = rdf_func(trajectory_model_np[t_for_rdf], box_size, dr=0.1)

# Plot RDF
plt.plot(radii_model, g_r_model)#, label='Model Data 1')
plt.xlabel('Distance')
plt.ylabel('Radial Distribution Function (RDF)')
plt.legend()
plt.savefig('results_rdfgraphs_velocity/rdf_graph_both_333_fin.png')

print("Finished")

plt.close()


## an average of multiple snapshots

t_for_rdf = 3
t_for_rdf_end = 89

g_r_real_all = []
g_r_model_all = []

trajectory_real_np = np.mod(trajectory_real_np,BOX_SIZE)

for i in range (t_for_rdf, t_for_rdf_end):
    # Calculate RDF with periodic boundary conditions
    g_r_real, radii_real = rdf_func(trajectory_real_np[i], box_size, dr=0.1)
    g_r_real_all.append(g_r_real)


g_r_real_average = np.mean(g_r_real_all, axis=0)

# Plot RDF
plt.plot(radii_real, g_r_real_average, label='Real Data')
plt.xlabel('Distance')
plt.ylabel('Average Radial Distribution Function (RDF)')
plt.legend()
#plt.savefig('rdf_graph_real.png')



trajectory_model_np = np.mod(trajectory_model_np,BOX_SIZE)

for j in range (t_for_rdf, t_for_rdf_end):
    # Calculate RDF with periodic boundary conditions
    g_r_model, radii_model = rdf_func(trajectory_model_np[j], box_size, dr=0.1)
    g_r_model_all.append(g_r_model)

g_r_model_average = np.mean(g_r_model_all, axis=0)

# Plot RDF
plt.plot(radii_model, g_r_model_average, label='Model Data')
plt.xlabel('Distance')
plt.ylabel('Average Radial Distribution Function (RDF)')
plt.legend()
plt.savefig('results_rdfgraphs_velocity/rdf_graph_average_fin.png')

print("Finished")

### visualize the trajectory of one particle

particle_n = 88

time_final = 99

trajectory_of1_real = trajectory_real_np[0:time_final,particle_n,:]
print("Size of trjectory (should be len(t)*3)")
print(trajectory_of1_real.size)
trajectory_of1_model = trajectory_model_np[0:time_final:,particle_n,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_of1_real[:, 0], trajectory_of1_real[:, 1], trajectory_of1_real[:, 2], label=f'Particle {particle_n + 1}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Trajectory of Particle (one) {particle_n + 1} over Time + vel')
ax.legend()
fig.savefig(f"Trjectory of {particle_n + 1} (real)")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_of1_model[:, 0], trajectory_of1_model[:, 1], trajectory_of1_model[:, 2], label=f'Particle {particle_n + 1}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Trajectory of Particle (one) {particle_n + 1} over Time + vel')
ax.legend()
fig.savefig(f"Trjectory of {particle_n + 1} (model)")

print(trajectory_of1_model)
print(trajectory_of1_real)

particle_n = 3

time_final = 5

trajectory_of1_real = trajectory_real_np[0:time_final,particle_n,:]
print("Size of trjectory (should be len(t)*3)")
print(trajectory_of1_real.size)
trajectory_of1_model = trajectory_model_np[0:time_final:,particle_n,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_of1_real[:, 0], trajectory_of1_real[:, 1], trajectory_of1_real[:, 2], label=f'Particle {particle_n + 1}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Trajectory of Particle (one) {particle_n + 1} over Time + vel')
ax.legend()
fig.savefig(f"Trjectory of {particle_n + 1} (real)")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_of1_model[:, 0], trajectory_of1_model[:, 1], trajectory_of1_model[:, 2], label=f'Particle {particle_n + 1}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Trajectory of Particle (one) {particle_n + 1} over Time + vel')
ax.legend()
fig.savefig(f"Trjectory of {particle_n + 1} (model)")

print(trajectory_of1_model)
print(trajectory_of1_real)

particle_n = 106

time_final = 5

trajectory_of1_real = trajectory_real_np[0:time_final,particle_n,:]
print("Size of trjectory (should be len(t)*3)")
print(trajectory_of1_real.size)
trajectory_of1_model = trajectory_model_np[0:time_final:,particle_n,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_of1_real[:, 0], trajectory_of1_real[:, 1], trajectory_of1_real[:, 2], label=f'Particle {particle_n + 1}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Trajectory of Particle (one) {particle_n + 1} over Time + vel')
ax.legend()
fig.savefig(f"Trjectory of {particle_n + 1} (real)")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_of1_model[:, 0], trajectory_of1_model[:, 1], trajectory_of1_model[:, 2], label=f'Particle {particle_n + 1}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Trajectory of Particle (one) {particle_n + 1} over Time + vel')
ax.legend()
fig.savefig(f"Trjectory of {particle_n + 1} (model)")

print(trajectory_of1_model)
print(trajectory_of1_real)

particle_n = 234

time_final = 5

trajectory_of1_real = trajectory_real_np[0:time_final,particle_n,:]
print("Size of trjectory (should be len(t)*3)")
print(trajectory_of1_real.size)
trajectory_of1_model = trajectory_model_np[0:time_final:,particle_n,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_of1_real[:, 0], trajectory_of1_real[:, 1], trajectory_of1_real[:, 2], label=f'Particle {particle_n + 1}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Trajectory of Particle (one) {particle_n + 1} over Time + vel')
ax.legend()
fig.savefig(f"Trjectory of {particle_n + 1} (real)")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_of1_model[:, 0], trajectory_of1_model[:, 1], trajectory_of1_model[:, 2], label=f'Particle {particle_n + 1}')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'Trajectory of Particle (one) {particle_n + 1} over Time + vel')
ax.legend()
fig.savefig(f"Trjectory of {particle_n + 1} (model)")

print(trajectory_of1_model)
print(trajectory_of1_real)