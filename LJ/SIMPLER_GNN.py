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
import math

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

from LATENT_ODE.create_latent_ode_model import *
from torch.distributions.normal import Normal
import LATENT_ODE.utils as utils

import torch.nn as nn

from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus
from dgl.nn import CFConv, NNConv
from dgl.nn.pytorch.factory import RadiusGraph

def cubic_kernel(r, re):
    eps = 1e-3
    r = torch.threshold(r, eps, re)
    return nn.ReLU()((1. - (r/re)**2)**3)


class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim=128,
                 hidden_layer=3,
                 activation_first=False,
                 activation='relu',
                 init_param=False):
        super(MLP, self).__init__()
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'softplus':
            act_fn = nn.Softplus()
        elif activation == 'shifted_softplus':
            act_fn = nn.ShiftedSoftplus()
        else:
            raise Exception('Only support: relu, leaky_relu, sigmoid, tanh, elu, as non-linear activation')

        mlp_layer = []
        for l in range(hidden_layer):
            if l != hidden_layer-1 and l != 0:
                mlp_layer += [nn.Linear(hidden_dim, hidden_dim), act_fn]
            elif l == 0:
                if hidden_layer == 1:
                    if activation_first:
                        mlp_layer += [act_fn, nn.Linear(in_feats, out_feats)]
                    else:
                        print('Using MLP with no hidden layer and activations! Fall back to nn.Linear()')
                        mlp_layer += [nn.Linear(in_feats, out_feats)]
                elif not activation_first:
                    mlp_layer += [nn.Linear(in_feats, hidden_dim), act_fn]
                else:
                    mlp_layer += [act_fn, nn.Linear(in_feats, hidden_dim), act_fn]
            else:   # l == hidden_layer-1
                mlp_layer += [nn.Linear(hidden_dim, out_feats)]
        self.mlp_layer = nn.Sequential(*mlp_layer)
        if init_param:
            self._init_parameters()

    def _init_parameters(self):
        for layer in self.mlp_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, feat):
        return self.mlp_layer(feat)
    

class SmoothConvBlockNew(nn.Module):
    def __init__(self,
                 in_node_feats,
                 out_node_feats,
                 hidden_dim=128,
                 conv_layer=2,  # at some point try with 3
                 edge_emb_dim=1,
                 use_layer_norm=False,
                 use_batch_norm=False, ##changed it
                 drop_edge=False,
                 activation='relu',
                 update_egde_emb=False,
                 ):
        super(SmoothConvBlockNew, self).__init__()
        self.conv = nn.ModuleList()
        self.edge_emb_dim = edge_emb_dim
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm

        self.drop_edge = drop_edge
        if use_batch_norm == use_layer_norm and use_batch_norm:
            raise Exception('Only one type of normalization at a time')
        if use_layer_norm or use_batch_norm:
            self.norm_layers = nn.ModuleList()

        self.edge_func = MLP(1, in_node_feats*out_node_feats)

        network = NNConv(in_node_feats, out_node_feats, self.edge_func)
        #network = CFConv(in_node_feats, 1, hidden_dim, out_node_feats)

        for layer in range(conv_layer):
            if layer == 0:
                self.conv.append(network)
            else:
                self.conv.append(network)
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(out_node_feats))
            elif use_batch_norm:
                self.norm_layers.append(nn.BatchNorm1d(out_node_feats)) 

    def forward(self, graph: dgl.DGLGraph, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:

        for l, conv_layer in enumerate(self.conv):
            if self.use_layer_norm or self.use_batch_norm:
                h = conv_layer.forward(graph, self.norm_layers[l](h), e) + h
            else:
                h = conv_layer.forward(graph, h, e) + h

        return h


# code from DGL documents
class RBFExpansion(nn.Module):
    r"""Expand distances between nodes by radial basis functions.

    .. math::
        \exp(- \gamma * ||d - \mu||^2)

    where :math:`d` is the distance between two nodes and :math:`\mu` helps centralizes
    the distances. We use multiple centers evenly distributed in the range of
    :math:`[\text{low}, \text{high}]` with the difference between two adjacent centers
    being :math:`gap`.

    The number of centers is decided by :math:`(\text{high} - \text{low}) / \text{gap}`.
    Choosing fewer centers corresponds to reducing the resolution of the filter.

    Parameters
    ----------
    low : float
        Smallest center. Default to 0.
    high : float
        Largest center. Default to 30.
    gap : float
        Difference between two adjacent centers. :math:`\gamma` will be computed as the
        reciprocal of gap. Default to 0.1.
    """
    def __init__(self, low=0., high=30., gap=0.1):
        super(RBFExpansion, self).__init__()

        num_centers = int(np.ceil((high - low) / gap))
        self.centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(), requires_grad=False)
        self.gamma = 1 / gap

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(
            self.centers.clone().detach().float(), requires_grad=False).to(device)

    def forward(self, edge_dists):
        """Expand distances.

        Parameters
        ----------
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (E, len(self.centers))
            Expanded distances.
        """
        radial = edge_dists - self.centers
        coef = - self.gamma
        return torch.exp(coef * (radial ** 2))


class Encoder(nn.Module):  # no bond, no learnable node encoder
    def __init__(self,
                encoding_size,
                out_feats,
                box_size,   # can also be array
                hidden_dim=128,
                conv_layer=2,
                edge_embedding_dim=32,
                dropout=0.1,
                drop_edge=True,
                use_layer_norm=False):
        super(Encoder, self).__init__()

        self.graph_conv = SmoothConvBlockNew(in_node_feats=encoding_size,
                                             out_node_feats=encoding_size,
                                             hidden_dim=hidden_dim,
                                             conv_layer=conv_layer,
                                             edge_emb_dim=1,
                                             use_layer_norm=use_layer_norm,
                                             use_batch_norm=not use_layer_norm,
                                             drop_edge=drop_edge,
                                             activation='silu')

        self.edge_emb_dim = edge_embedding_dim
        self.edge_expand = RBFExpansion(high=1, gap=0.025)
        self.edge_drop_out = nn.Dropout(dropout)

        self.length_mean = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.length_std = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.length_scaler = StandardScaler()

        if isinstance(box_size, np.ndarray):
            self.box_size = torch.from_numpy(box_size).float()
        else:
            self.box_size = box_size
        self.box_size = self.box_size
        half_ec = int(encoding_size/2)

        self.node_encoder = MLP(3, half_ec, hidden_layer=2, hidden_dim=128, activation='leaky_relu')
        self.vel_encoder = MLP(3, half_ec, hidden_layer=2, hidden_dim=128, activation='leaky_relu')
        self.MLPbig_to_small = MLP(encoding_size*258, out_feats, hidden_layer=2, hidden_dim=128, activation='leaky_relu')


        # with TAGConv
        self.norm = nn.LayerNorm(encoding_size)

    def build_graph(self,
                    fluid_edge_idx: torch.Tensor,
                    fluid_pos: torch.Tensor,
                    fluid_vel: torch.Tensor,
                    self_loop=True) -> dgl.DGLGraph:

        center_idx = fluid_edge_idx[0, :]  # [edge_num, 1]
        neigh_idx = fluid_edge_idx[1, :]
        fluid_graph = dgl.graph((neigh_idx, center_idx))

        # Get the positions of the nodes forming each edge
        start_pos = fluid_pos[center_idx]
        end_pos = fluid_pos[neigh_idx]

        # Calculate Euclidean distances between nodes for each edge
        distances = torch.sqrt(torch.sum((start_pos - end_pos) ** 2, dim=1))

        c = torch.max(distances)

        # Add distances as edge weights to the graph
        fluid_graph.edata['w'] = (1/2)*(torch.cos(distances*math.pi/c)+1)

        #add node embeddings
        positions_encoded = self.node_encoder(fluid_pos)
        velocities_encoded = self.vel_encoder(fluid_vel)
        
        new_node = torch.cat((positions_encoded, velocities_encoded), axis=1)
        fluid_graph.ndata['e'] = new_node

        # add self loop for fluid particles
        if self_loop:
            fluid_graph.add_self_loop()
            #graph_to_save.add_self_loop()

        return fluid_graph

    def build_graph_batches(self, pos_lst, edge_idx_lst, vel_lst):
        graph_lst = []
        for pos, edge_idx, vel in zip(pos_lst, edge_idx_lst, vel_lst):
            graph = self.build_graph(edge_idx, pos, vel)
            graph_lst += [graph]
        batched_graph = dgl.batch(graph_lst)
        return batched_graph

    def _update_length_stat(self, new_mean, new_std):
        self.length_mean[0] = new_mean[0]
        self.length_std[0] = new_std[0]

    def fit_length(self, length):
        if not isinstance(length, np.ndarray):
            length = length.detach().cpu().numpy().reshape(-1, 1)
        self.length_scaler.partial_fit(length)

    def gencoder_mine(self, h: torch.Tensor, e: torch.Tensor, g: dgl.DGLGraph):
        out = self.graph_conv(g, h, e.reshape(-1, 1))
        out = self.norm(out)
        d1, d3 = out.size()
        d2 = 258
        d1 = int(d1/d2)
        out = out.view(d1, d2*d3)
        out = self.MLPbig_to_small(out)
        return out

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                fluid_edge_lst: List[torch.Tensor],
                fluid_vel_lst: List[torch.Tensor],
                ) -> torch.Tensor:
        if len(fluid_pos_lst) > 1:
            fluid_graph = self.build_graph_batches(fluid_pos_lst, fluid_edge_lst, fluid_vel_lst)
        else:
            fluid_graph = self.build_graph(fluid_edge_lst[0], fluid_pos_lst[0], fluid_vel_lst[0])

        g_embed = self.gencoder_mine(fluid_graph.ndata['e'], fluid_graph.edata['w'], fluid_graph)

        return g_embed

    def load_from_checkpoint(self, checkpoint_path):
        # Your loading logic here
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)
        return self

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class Decoder(nn.Module):
    def __init__(self,
                encoding_size,
                out_feats,
                box_size,   # can also be array
                hidden_dim=128,
                use_layer_norm=False):
        super(Decoder, self).__init__()

        self.decode_MLP = MLP(encoding_size, out_feats, hidden_layer=3, hidden_dim=256, activation='leaky_relu')
        self.decode_vel = MLP(encoding_size, out_feats, hidden_layer=3, hidden_dim=256, activation='leaky_relu')
        self.decode_forces = MLP(encoding_size, out_feats, hidden_layer=3, hidden_dim=256, activation='leaky_relu')
        
    def gdecoder_MLP(self, h: torch.Tensor):

        out = self.decode_MLP(h)
        vel = self.decode_vel(h)
        forces = self.decode_forces(h)
        return h, out, vel, forces

    def forward(self, h:torch.Tensor):

        h, out, vel, forces = self.gdecoder_MLP(h)
        return h, out, vel, forces

    def load_from_checkpoint(self, checkpoint_path):
        # Your loading logic here
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)
        return self
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

class Autoencoder(nn.Module):
    def __init__(self,
                encoding_size,
                out_feats,
                box_size,   # can also be array
                hidden_dim=128,
                conv_layer=4,
                edge_embedding_dim=32,
                dropout=0.1,
                drop_edge=True,
                use_layer_norm=False,):
        super(Autoencoder, self).__init__()

        hidden_latentode_size = 512

        self.encoder = Encoder(encoding_size,
                    hidden_latentode_size,
                    box_size,   # can also be array
                    hidden_dim=128,
                    conv_layer=4,
                    edge_embedding_dim=32,
                    dropout=0.1,
                    drop_edge=True,
                    use_layer_norm=False)

        self.decoder = Decoder(
                    hidden_latentode_size,
                    out_feats*258,
                    box_size,   # can also be array
                    hidden_dim=128,
                    use_layer_norm=False)

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                fluid_edge_lst: List[torch.Tensor],
                fluid_vel_lst: List[torch.Tensor],
                ):

        embed = self.encoder(fluid_pos_lst, fluid_edge_lst, fluid_vel_lst)

        h, pos_result, vel_result, forces_result = self.decoder(embed)

        d1, d2 = pos_result.size()

        pos_result = pos_result.view(d1, 258, 3)
        vel_result = vel_result.view(d1, 258, 3)

        return pos_result, vel_result, forces_result

class Autoencoder2(nn.Module):
    def __init__(self,
                encoding_size,
                out_feats,
                box_size,   # can also be array
                cutoff_radius,
                hidden_dim=128,
                conv_layer=4,
                edge_embedding_dim=32,
                dropout=0.1,
                drop_edge=True,
                use_layer_norm=False,):
        super(Autoencoder2, self).__init__()

        hidden_latentode_size = 512

        self.encoder = Encoder(encoding_size,
                    hidden_latentode_size,
                    box_size,   # can also be array
                    hidden_dim=128,
                    conv_layer=4,
                    edge_embedding_dim=32,
                    dropout=0.1,
                    drop_edge=True,
                    use_layer_norm=False)

        self.decoder = Decoder(
                    hidden_latentode_size,
                    out_feats*258,
                    box_size,   # can also be array
                    hidden_dim=128,
                    use_layer_norm=False)
    
        self.rg = RadiusGraph(cutoff_distance)

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],
                ):

        graph = self.make_a_graph(fluid_pos_lst)

        embed = self.encoder(fluid_pos_lst, fluid_edge_lst, fluid_vel_lst)

        h, pos_result, vel_result, forces_result = self.decoder(embed)

        d1, d2 = pos_result.size()

        pos_result = pos_result.view(d1, 258, 3)
        vel_result = vel_result.view(d1, 258, 3)

        return pos_result, vel_result, forces_result

    def make_a_graph(self, fluid_pos_lst, cutoff_distance):
        graph, dist = rg(torch.stack(fluid_pos_lst), get_distances=True)
        graph.edata['w'] = (1/2)*(torch.cos(math.pi*dist/cutoff_distance)+1)
        graph.ndata['e'] = torch.stack(fluid_pos_lst)


