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
from train_AUTOENCODER import *

num_particles = 258
BOX_SIZE = 27.27
box_size = np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])

def autoencode(start_pos, start_vel, cp_name, epoch):
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
                        loss='mse')
    model = ParticleAutoencoder(args).load_from_checkpoint(PATH, args=args)
    model.load_training_stats(SCALER_CKPT)
    model.cuda()
    model.eval()

    with torch.no_grad():

        autoencoded = model.autoencode(start_pos, start_vel)

    return autoencoded

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
def rdf_graph_one_snapshot(pos_real, pos_encoded, directory, epoch):

    # Calculate RDF with periodic boundary conditions
    g_r_real, radii_real = rdf_func(pos_real, box_size, dr=0.1)

    # Plot RDF
    plt.plot(radii_real, g_r_real, label='Real Data')
    plt.xlabel('Distance')
    plt.ylabel('Radial Distribution Function (RDF)')
    plt.legend()
    #plt.savefig('rdf_graph_real.png')

    pos_encoded = np.mod(pos_encoded,BOX_SIZE)

    g_r_model, radii_model = rdf_func(pos_encoded, box_size, dr=0.1)

    # Plot RDF
    plt.plot(radii_model, g_r_model, label='Model Data')
    plt.xlabel('Distance')
    plt.ylabel('Radial Distribution Function (RDF)')
    plt.legend()
    plt.savefig(f"{directory}/rdfgraph.png")
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

def test_main(args):
    cp_name = args.cp_name
    epoch = args.epoch

    directory = os.path.join('results_autoencoder',f'results_{cp_name}_epoch={epoch}')
    os.makedirs(directory, exist_ok=True)

    ## for training data
    everything_train = np.load(f'md_dataset/lj_data/data_3_678.npz')
    just_pos_train = everything_train['pos']
    just_vel_train = everything_train['vel']

    autoencoded_train = autoencode(just_pos_train, just_vel_train, cp_name, epoch)

    autoencoded_train = np.mod(autoencoded_train,BOX_SIZE)

    pos_tensor_train = torch.tensor(just_pos_train)
    autoencoded_tensor_train = torch.tensor(autoencoded_train)

    rdf_graph_one_snapshot(just_pos_train, autoencoded_train, f'{cp_name}_train', epoch)

    loss_train = nn.MSELoss()(pos_tensor_train,autoencoded_tensor_train)

    print("Train loss is:")
    print(loss_train)

    ## save trajectory data
    file_path = os.path.join(directory, f'1_time_positions_train')
    np.save(file_path, autoencoded_train)

    print(pos_tensor_train)
    print(autoencoded_train)

    ## for test data
    everything_test = np.load(f'md_dataset/lj_data_to_test/data_0_678.npz')
    just_pos_test = everything_test['pos']
    just_vel_test = everything_test['vel']

    autoencoded_test = autoencode(just_pos_test, just_vel_test, cp_name, epoch)

    autoencoded_test = np.mod(autoencoded_test, BOX_SIZE)

    pos_tensor_test = torch.tensor(just_pos_test)
    autoencoded_tensor_test = torch.tensor(autoencoded_test)

    rdf_graph_one_snapshot(just_pos_test, autoencoded_test, f'{cp_name}_test', epoch)

    loss_test = nn.MSELoss()(pos_tensor_test,autoencoded_test)

    print("Test loss is:")
    print(loss_test)

    ## save trajectory data
    file_path = os.path.join(directory, f'1_time_positions_test')
    np.save(file_path, autoencoded_test)

    print(pos_tensor_test)
    print(autoencoded_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp_name', default='AUTOENCODER_50ts(cords)_NNConv_512')
    parser.add_argument('--epoch', default=5, type=int)
    args = parser.parse_args()
    test_main(args)


if __name__ == '__main__':
    main()