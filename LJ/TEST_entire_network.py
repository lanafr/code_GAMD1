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

import torch
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn

from einops import rearrange
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from train_ENTIRE_NETWORK import *

num_particles = 258
BOX_SIZE = 27.27
box_size = np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])

def network_trajectory(start_pos, start_vel, t, architecture1, cp_name, epoch):
    PATH = f'/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/{cp_name}/checkpoint_{epoch}.ckpt'
    SCALER_CKPT = f'/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/{cp_name}/scaler_{epoch}.npz'
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
                        mode = 'test',
                        architecture = architecture1,
                        loss='mse')
    model = MDSimNet(args).load_from_checkpoint(PATH, args=args)
    model.load_training_stats(SCALER_CKPT)
    model.cuda()
    model.eval()

    with torch.no_grad():

        trajectory = model.run_the_network(start_pos, start_vel, t)
        print(model.get_temp_pairs(torch.tensor(trajectory)))

    return trajectory

def rdf_func(coords, box_size, dr=0.1, r_max=None):
    num_particles = len(coords)
    if r_max is None:
        r_max = np.min(box_size) / 2.0  # Use half the minimum box size as a default

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

## for one time snapshot
def rdf_graph_one_snapshot(t,trajectory_real_np, trajectory_model_np, directory, epoch):

    # Calculate RDF with periodic boundary conditions
    g_r_real, radii_real = rdf_func(trajectory_real_np[t], box_size, dr=0.1)

    # Plot RDF
    plt.plot(radii_real, g_r_real, label='Real Data')
    plt.xlabel('Distance')
    plt.ylabel('Radial Distribution Function (RDF)')
    plt.legend()
    #plt.savefig('rdf_graph_real.png')

    trajectory_model_np = np.mod(trajectory_model_np,BOX_SIZE)

    g_r_model, radii_model = rdf_func(trajectory_model_np[t], box_size, dr=0.1)

    # Plot RDF
    plt.plot(radii_model, g_r_model)#, label='Model Data 1')
    plt.xlabel('Distance')
    plt.ylabel('Radial Distribution Function (RDF)')
    plt.legend()
    plt.savefig(f"{directory}/rdfgraph_(t={t})_(epoch={epoch}).png")
    plt.close()


## an average of multiple snapshots
def rdf_graph_multiple_snapshots(t_start, t_end, trajectory_real_np, trajectory_model_np, directory, epoch):
    g_r_real_all = []
    g_r_model_all = []

    trajectory_real_np = np.mod(trajectory_real_np,BOX_SIZE)

    for i in range (t_start, t_end):
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

    for j in range (t_start, t_end):
        # Calculate RDF with periodic boundary conditions
        g_r_model, radii_model = rdf_func(trajectory_model_np[j], box_size, dr=0.1)
        g_r_model_all.append(g_r_model)

    g_r_model_average = np.mean(g_r_model_all, axis=0)

    # Plot RDF
    plt.plot(radii_model, g_r_model_average, label='Model Data')
    plt.xlabel('Distance')
    plt.ylabel('Average Radial Distribution Function (RDF)')
    plt.legend()
    plt.xlim(0,14)
    plt.ylim(0,0.05)

    plt.savefig(f"{directory}/rdfgraph_(t={t_start}-{t_end})_(epoch={epoch}).png")
    plt.close()


### visualize the trajectory of one particle
def plot_of_trajectory(particle_num, time_start, time_final, trajectory_real_np, trajectory_model_np, directory):

    trajectory_of1_real = trajectory_real_np[time_start:time_final,particle_num,:]
    trajectory_of1_model = trajectory_model_np[time_start:time_final,particle_num,:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory_of1_real[:, 0], trajectory_of1_real[:, 1], trajectory_of1_real[:, 2], label=f'Real Data, Particle {particle_num + 1}')
    ax.plot(trajectory_of1_model[:, 0], trajectory_of1_model[:, 1], trajectory_of1_model[:, 2], label=f'Model Data, Particle {particle_num + 1}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'Trajectory of Particle {particle_num + 1} over Time')
    ax.legend()
    fig.savefig(f"{directory}/Trajectory of {particle_num + 1} from t={time_start} to t={time_final}")

def save_xyz_from_numpy(coordinates, output_file):
    num_frames = coordinates.shape[0]
    num_atoms = coordinates.shape[1]

    with open(output_file, 'w') as f:
        for frame_idx in range(num_frames):
            f.write(str(num_atoms) + '\n')
            f.write("\n")
            for atom_idx in range(num_atoms):
                atom = coordinates[frame_idx, atom_idx]
                f.write(f"Ar {atom[0]} {atom[1]} {atom[2]}\n")

def test_main(args):
    t_from = args.t_from
    t_how_many = args.t_how_many
    t_rdf_one_snapshot = args.t_rdf_one_snapshot
    t_rdf_multiple_snapshots_start = args.t_rdf_multiple_snapshots_start
    t_rdf_multiple_snapshots_end = args.t_rdf_multiple_snapshots_end
    cp_name = args.cp_name
    epoch = args.epoch
    architecture = args.architecture

    directory = os.path.join('results_all',f'results_{cp_name}_epoch={epoch}')
    os.makedirs(directory, exist_ok=True)

    start_pos = []
    start_vel = []

    if architecture == 'node' or architecture == 'recurrent':
        start_all = np.load(f'md_dataset/lj_data_20ts/data_0_{t_from}.npz')
        start_pos = start_all['pos']
        start_vel = start_all['vel']

    if architecture == 'latentode' or architecture == 'graphlatentode':
        for i in range (t_from,t_from+int(t_how_many/2)):
            start_all = np.load(f'md_dataset/lj_data_20ts/data_0_{i}.npz')
            start_pos.append(start_all['pos'])
            start_vel.append(start_all['vel'])

    trajectory_real = []
    trajectory_model = []

    for i in range(t_from,t_from+int(t_how_many/2)): ## changed for graph latent odes
        everything = np.load(f'md_dataset/lj_data_20ts/data_0_{i}.npz')
        just_pos = everything['pos']
        trajectory_real.append(just_pos)

    trajectory_model = network_trajectory(start_pos, start_vel, t_how_many, architecture, cp_name, epoch)

    trajectory_real_np= np.stack(trajectory_real, axis=0)
    trajectory_model_np = trajectory_model

    trajectory_model_np = np.mod(trajectory_model_np,BOX_SIZE)

    trajectory_real_tensor = torch.tensor(trajectory_real_np)
    trajectory_model_tensor = torch.tensor(trajectory_model_np)

    loss = nn.MSELoss()(trajectory_real_tensor,trajectory_model_tensor)

    print("Loss is:")
    print(loss)

    print("Difference is:")
    print(trajectory_model_np-trajectory_real_np)

    print()

    ## plot rdf graphs
    rdf_graph_one_snapshot(0, trajectory_real_np, trajectory_model_np, directory, epoch)
    rdf_graph_one_snapshot(t_rdf_one_snapshot, trajectory_real_np, trajectory_model_np, directory, epoch)
    rdf_graph_multiple_snapshots(t_rdf_multiple_snapshots_start, int(t_rdf_multiple_snapshots_end/2), trajectory_real_np, trajectory_model_np, directory, epoch)

    ## plot particle trajectories
    
    plot_of_trajectory(10, t_rdf_multiple_snapshots_start, t_rdf_multiple_snapshots_end, trajectory_real_np, trajectory_model_np, directory)
    plot_of_trajectory(50, t_rdf_multiple_snapshots_start, t_rdf_multiple_snapshots_end, trajectory_real_np, trajectory_model_np, directory)
    plot_of_trajectory(88, t_rdf_multiple_snapshots_start, t_rdf_multiple_snapshots_end, trajectory_real_np, trajectory_model_np, directory)
    plot_of_trajectory(100, t_rdf_multiple_snapshots_start, t_rdf_multiple_snapshots_end, trajectory_real_np, trajectory_model_np, directory)
    plot_of_trajectory(199, t_rdf_multiple_snapshots_start, t_rdf_multiple_snapshots_end, trajectory_real_np, trajectory_model_np, directory)

    ## save trajectory data
    file_path = os.path.join(directory, f'trajectory_data_t=0-{t_how_many}.xyz')
    #np.save(file_path, trajectory_model_np)
    save_xyz_from_numpy(trajectory_model_np, file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t_from', default = 1100, type=int)
    parser.add_argument('--t_how_many', default = 20, type=int)
    parser.add_argument('--t_rdf_one_snapshot', default=9, type=int)
    parser.add_argument('--t_rdf_multiple_snapshots_start', default=0, type=int)  
    parser.add_argument('--t_rdf_multiple_snapshots_end', default=19, type=int)   
    parser.add_argument('--cp_name', default='ENTIRE_NETWORK_emb+cord')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--architecture', default='node', type=str)
    args = parser.parse_args()
    test_main(args)


if __name__ == '__main__':
    main()