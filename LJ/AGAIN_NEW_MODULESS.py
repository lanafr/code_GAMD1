###########################
# Encoder part of this code was partially taken from GAMD (https://arxiv.org/abs/2112.03383, Li, Zijie and Meidani, Kazem and Yadav, Prakarsh and Barati Farimani, Amir, 2022.)
###########################

import numpy as np
import torch
import torch.nn as nn
import dgl.nn
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn import GraphConv
import dgl.function as fn
import time
from md_module import get_neighbor
from sklearn.preprocessing import StandardScaler
import random
import os, sys

from typing import List, Set, Dict, Tuple, Optional

from dgl.nn import NNConv
from dgl.nn import TAGConv
from egnn_pytorch import EGNN

import argparse
import os, sys
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import jax
import jax.numpy as jnp
import cupy
from pytorch_lightning.loggers import WandbLogger
from torchdiffeq import odeint, odeint_adjoint

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
#from nn_module import SimpleMDNetNew
from train_utils_seq import Sequential_data, Some_seq_data, just_a_sequence
from graph_utils import NeighborSearcher, graph_network_nbr_fn
import time

import torchcde

import wandb

import torch.utils.data as data

from GRAPH_LATENT_ODE.lib.create_coupled_ode_model import *
from torch.distributions.normal import Normal
import GRAPH_LATENT_ODE.lib.utils as utils
from torch_geometric.data import Data, Batch
import math
from scipy.linalg import block_diag

def cubic_kernel(r, re):
    eps = 1e-3
    r = torch.threshold(r, eps, re)
    return nn.ReLU()((1. - (r/re)**2)**3)

class Args_latentODE:
    def __init__(self):
        self.output_dim = 3
        self.ode_dims = 128 #"Dimensionality of the ODE func for edge and node (must be the same)"
        self.rec_dims = 128 #"Dimensionality of the recognition model ."
        self.augment_dim = 0 #'augmented dimension'
        self.rec_layers = 3 #"Number of layers in recognition model "
        self.z0_encoder = 'GTrans' 
        self.dropout = 0.2
        self.num_atoms = 258
        self.solver = "rk4"

def create_mask(x, percentage):
    mask = torch.ones_like(x)
    num_timesteps = x.size()[0]
    num_masked_timesteps = int(percentage * num_timesteps)
    masked_indices = np.sort(np.random.permutation(num_timesteps)[:num_masked_timesteps], axis=None)
    mask[masked_indices,:] = 0

    not_masked_indices = np.ones(num_timesteps)
    not_masked_indices[masked_indices] = False
    not_masked_indices = np.sort(not_masked_indices, axis=None)

    return mask, masked_indices, not_masked_indices

class latentODE(nn.Module):

    def __init__(self, encoding_size, mode):
        super(latentODE, self).__init__()
        
        args = Args_latentODE()
        device = "cuda"
        input_dim = encoding_size
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        obsrv_std = 0.01
        obsrv_std = torch.Tensor([obsrv_std]).to(device)

        self.CoupledODE = create_CoupledODE_model(args, input_dim, z0_prior, obsrv_std, device)
        self.mode = mode

    def create_tensor(self, N, K):
        tensor_list = []
        for i in range(K + 1):
            tensor_list.append(torch.full((N,), i))
        return torch.cat(tensor_list)

    def forward(self, data, t):
        data.batch = torch.cat([torch.full((258,), i) for i in range(int(t/2))]).cuda()
        timesteps_to_predict = torch.arange(int(t/2),t).float().cuda()

        pred_node, pred_edges, all_extra_info, temporal_weights = self.CoupledODE.get_reconstruction(data,
                                            timesteps_to_predict,
                                            258)

        return pred_node, pred_edges

class EntireModel(nn.Module):
    def __init__(self,
                encoding_size,
                out_feats,
                box_size,   # can also be array
                mode,
                architecture,
                hidden_dim=128,
                conv_layer=2,
                edge_embedding_dim=32,
                dropout=0.1,
                drop_edge=True,
                use_layer_norm=False):
        super(EntireModel, self).__init__()

        hidden_latentode_size = 128

        self.encoding_size = encoding_size

        self.latentODE = latentODE(3, mode)

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                fluid_edge_lst: List[torch.Tensor],
                fluid_vel_lst: List[torch.Tensor],
                t,
                ):

        feature = fluid_pos_lst.permute(1, 0, 2)
        
        c = 1

        edges_lst = []

        time = np.arange(0,t)
        for i in range (t):
            edges_lst.append(self.get_edge_weight_matrix(fluid_pos_lst[i], fluid_edge_lst[i], c))
        edge = torch.stack(edges_lst)

        graph_data_1, edge_num_1 = self.make_one(feature.detach().cpu().numpy(), edge.detach().cpu().numpy(), time)

        pred_node, pred_edges = self.latentODE(graph_data_1, t*2)

        return pred_node, pred_edges

    def get_edge_weight(self,
                        fluid_pos,
                        fluid_edge_idx,
                        c):

        center_idx = fluid_edge_idx[0, :]  # [edge_num, 1]
        neigh_idx = fluid_edge_idx[1, :]

        # Get the positions of the nodes forming each edge
        start_pos = fluid_pos[center_idx]
        end_pos = fluid_pos[neigh_idx]

        # Calculate Euclidean distances between nodes for each edge
        distances = torch.sqrt(torch.sum((start_pos - end_pos) ** 2, dim=1))

        # Add distances as edge weights to the graph
        return (1/2)*(torch.cos(distances*math.pi/c)+1)

    def get_edge_weight_matrix(self, fluid_pos, fluid_edge_idx, c):
        N = fluid_pos.shape[0]  # Number of nodes, derived from fluid_pos
        
        # Extract the start and end indices of each edge
        center_idx = fluid_edge_idx[0, :]  # Start node indices of edges
        neigh_idx = fluid_edge_idx[1, :]   # End node indices of edges

        # Get the positions of the nodes forming each edge
        start_pos = fluid_pos[center_idx]
        end_pos = fluid_pos[neigh_idx]

        # Calculate Euclidean distances between nodes for each edge
        distances = torch.sqrt(torch.sum((start_pos - end_pos) ** 2, dim=1))

        # Calculate edge weights
        edge_weights = (1/2) * (torch.cos(distances * math.pi / c) + 1)

        # Initialize a matrix of zeros for all possible edges
        edge_weight_matrix = torch.zeros(N, N)
        
        # Use the indices to update the matrix with the calculated edge weights
        # This loop iterates over each edge and updates the matrix
        for i, (start, end) in enumerate(zip(center_idx, neigh_idx)):
            edge_weight_matrix[start, end] = edge_weights[i]

        return edge_weight_matrix

    def make_one(self, feature, edge, time):
        '''f

        :param feature: [N,T1,D]
        :param edge: [T,N,N]  (needs to transfer into [T1,N,N] first, already with self-loop)
        :param time: [T1]
        :return:
            1. x : [N*T1,D]: feature for each node.
            2. edge_index [2,num_edge]: edges including cross-time
            3. edge_weight [num_edge]: edge weights
            4. y: [N], value= num_steps: number of timestamps for each state node.
            5. x_pos 【N*T1】: timestamp for each node
            6. edge_time [num_edge]: edge relative time.
        '''

        ########## Getting and setting hyperparameters:
        num_states = feature.shape[0]
        T1 = time.shape[0]
        each_gap = 1/ (time.shape[0]*2)
        time = np.reshape(time,(-1,1))

        ########## Compute Node related data:  x,y,x_pos
        # [Num_states],value is the number of timestamp for each state in the encoder, == args.condition_length
        y = time.shape[0]*np.ones(num_states)
        # [Num_states*T1,D]
        x = np.reshape(feature,(-1,feature.shape[2]))
        # [Num_states*T1,1] node timestamp
        x_pos = np.concatenate([time for i in range(num_states)],axis=0)
        assert len(x_pos) == feature.shape[0]*feature.shape[1]

        ########## Compute edge related data
        edge_time_matrix = np.concatenate([np.asarray(x_pos).reshape(-1, 1) for _ in range(len(x_pos))],
                                          axis=1) - np.concatenate(
            [np.asarray(x_pos).reshape(1, -1) for _ in range(len(x_pos))], axis=0)  # [N*T1,N*T1], SAME TIME = 0

        edge_exist_matrix = np.ones((len(x_pos), len(x_pos)))  # [N*T1,N*T1] NO-EDGE = 0, depends on both edge weight and time matrix
        print
        # Step1: Construct edge_weight_matrix [N*T1,N*T1]
        edge_repeat = np.repeat(edge, time.shape[0], axis=2)  # [T1,N,NT1]
        edge_repeat = np.transpose(edge_repeat, (1, 0, 2))  # [N,T1,NT1]
        edge_weight_matrix = np.reshape(edge_repeat, (-1, edge_repeat.shape[2]))  # [N*T1,N*T1]

        # mask out cross_time edges of different state nodes.
        a = np.identity(T1)  # [T,T]
        b = np.concatenate([a for i in range(num_states)], axis=0)  # [N*T,T]
        c = np.concatenate([b for i in range(num_states)], axis=1)  # [N*T,N*T]

        a = np.ones((T1, T1))
        d = block_diag(*([a] * num_states))
        edge_weight_mask = (1 - d) * c + d
        edge_weight_matrix = edge_weight_matrix * edge_weight_mask  # [N*T1,N*T1]

        max_gap = each_gap

        # Step2: Construct edge_exist_matrix [N*T1,N*T1]: depending on both time and weight.
        edge_exist_matrix = np.where(
            (edge_time_matrix <= 0) & (abs(edge_time_matrix) <= max_gap) & (edge_weight_matrix != 0),
            edge_exist_matrix, 0)


        edge_weight_matrix = edge_weight_matrix * edge_exist_matrix
        edge_index, edge_weight_attr = utils.convert_sparse(edge_weight_matrix)
        assert np.sum(edge_weight_matrix!=0)!=0  #at least one edge weight (one edge) exists.

        edge_time_matrix = (edge_time_matrix + 3) * edge_exist_matrix # padding 2 to avoid equal time been seen as not exists.
        _, edge_time_attr = utils.convert_sparse(edge_time_matrix)
        edge_time_attr -= 3

        # converting to tensor
        x = torch.FloatTensor(x).cuda()
        edge_index = torch.LongTensor(edge_index).cuda()
        edge_weight_attr = torch.FloatTensor(edge_weight_attr).cuda()
        edge_time_attr = torch.FloatTensor(edge_time_attr).cuda()
        y = torch.LongTensor(y).cuda()
        x_pos = torch.FloatTensor(x_pos).cuda()


        graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight_attr, y=y, pos=x_pos, edge_time = edge_time_attr)
        edge_num = edge_index.shape[1]

        return graph_data,edge_num