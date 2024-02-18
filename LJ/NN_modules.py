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
from nn_module import SimpleMDNetNew
from train_utils_seq import Sequential_data, Some_seq_data, just_a_sequence
from graph_utils import NeighborSearcher, graph_network_nbr_fn
import time

import torchcde

import wandb

import torch.utils.data as data

from LATENT_ODE.create_latent_ode_model import *
from torch.distributions.normal import Normal
import LATENT_ODE.utils as utils

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
        
class SmoothConvLayerNew(nn.Module):
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 hidden_dim=64,
                 activation='relu',
                 drop_edge=True,
                 #drop_edge=False,
                 update_edge_emb=False):

        super(SmoothConvLayerNew, self).__init__()
        self.drop_edge = drop_edge
        self.update_edge_emb = update_edge_emb
        if self.update_edge_emb:
            self.edge_layer_norm = nn.LayerNorm(in_edge_feats)

        # self.theta_src = nn.Linear(in_node_feats, hidden_dim)
        self.edge_affine = MLP(in_edge_feats, hidden_dim, activation=activation, hidden_layer=2)
        self.src_affine = nn.Linear(in_node_feats, hidden_dim)
        self.dst_affine = nn.Linear(in_node_feats, hidden_dim)
        self.theta_edge = MLP(hidden_dim, in_node_feats,
                              hidden_dim=hidden_dim, activation=activation, activation_first=True,
                              hidden_layer=2)
        # self.theta = MLP(hidden_dim, hidden_dim, activation_first=True, hidden_layer=2)

        self.phi_dst = nn.Linear(in_node_feats, hidden_dim)
        self.phi_edge = nn.Linear(in_node_feats, hidden_dim)
        self.phi = MLP(hidden_dim, out_node_feats,
                       activation_first=True, hidden_layer=1, hidden_dim=hidden_dim, activation=activation)

    def forward(self, g: dgl.DGLGraph, node_feat: torch.Tensor) -> torch.Tensor:
        h = node_feat.clone()
        with g.local_scope():
            if self.drop_edge and self.training:
                src_idx, dst_idx = g.edges()
                e_feat = g.edata['e'].clone()
                dropout_ratio = 0.2
                idx = np.arange(dst_idx.shape[0])
                np.random.shuffle(idx)
                keep_idx = idx[:-int(idx.shape[0] * dropout_ratio)]
                src_idx = src_idx[keep_idx]
                dst_idx = dst_idx[keep_idx]
                e_feat = e_feat[keep_idx]
                g = dgl.graph((src_idx, dst_idx))
                g.edata['e'] = e_feat
            # for multi batch training
            if g.is_block:
                h_src = h
                h_dst = h[:g.number_of_dst_nodes()]
            else:
                h_src = h_dst = h

            g.srcdata['h'] = h_src
            g.dstdata['h'] = h_dst
            edge_idx = g.edges()
            src_idx = edge_idx[0]
            dst_idx = edge_idx[1]
            edge_code = self.edge_affine(g.edata['e'])
            src_code = self.src_affine(h_src[src_idx])
            dst_code = self.dst_affine(h_dst[dst_idx])
            g.edata['e_emb'] = self.theta_edge(edge_code+src_code+dst_code)

            if self.update_edge_emb:
                normalized_e_emb = self.edge_layer_norm(g.edata['e_emb'])
            g.update_all(fn.src_mul_edge('h', 'e_emb', 'm'), fn.sum('m', 'h'))
            edge_emb = g.ndata['h']

        if self.update_edge_emb:
            g.edata['e'] = normalized_e_emb
        node_feat = self.phi(self.phi_dst(h) + self.phi_edge(edge_emb))
        return node_feat


class SmoothConvBlockNew(nn.Module):
    def __init__(self,
                 in_node_feats,
                 out_node_feats,
                 hidden_dim=128,
                 conv_layer=2,  # at some point try with 3
                 edge_emb_dim=32,
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

        for layer in range(conv_layer):
            if layer == 0:
                self.conv.append(SmoothConvLayerNew(in_node_feats=in_node_feats,
                                                 in_edge_feats=self.edge_emb_dim,
                                                 out_node_feats=out_node_feats,
                                                 hidden_dim=hidden_dim,
                                                 activation=activation,
                                                 drop_edge=drop_edge,
                                                 update_edge_emb=update_egde_emb))
            else:
                self.conv.append(SmoothConvLayerNew(in_node_feats=out_node_feats,
                                                 in_edge_feats=self.edge_emb_dim,
                                                 out_node_feats=out_node_feats,
                                                 hidden_dim=hidden_dim,
                                                 activation=activation,
                                                 drop_edge=drop_edge,
                                                 update_edge_emb=update_egde_emb))
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(out_node_feats))
            elif use_batch_norm:
                self.norm_layers.append(nn.BatchNorm1d(out_node_feats)) 

    def forward(self, h: torch.Tensor, graph: dgl.DGLGraph) -> torch.Tensor:

        for l, conv_layer in enumerate(self.conv):
            if self.use_layer_norm or self.use_batch_norm:
                h = conv_layer.forward(graph, self.norm_layers[l](h)) + h
            else:
                h = conv_layer.forward(graph, h) + h

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
                                             edge_emb_dim=edge_embedding_dim,
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
        #self.node_dencoder = MLP(64, 3, hidden_layer=2, hidden_dim=hidden_dim, activation='leaky_relu')
        self.edge_encoder = MLP(3 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
                                activation='leaky_relu')
        self.edge_layer_norm = nn.LayerNorm(self.edge_emb_dim)
        #self.graph_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')
        self.save_index = 0


        # with TAGConv


        #self.graph_e_1 = TAGConv(3, 256, activation=nn.ReLU())
        #self.graph_d_1 = TAGConv(256, 3, activation=nn.ReLU())
        #self.graph_e_2 = TAGConv(256, 128, activation=nn.GELU())
        #self.graph_d_2 = TAGConv(128, 256, activation=nn.GELU())
        #self.graphMLP1 = MLP(128, 64, activation_first=True, hidden_layer=3, hidden_dim=hidden_dim, activation = 'tanh')
        #self.graphMLP2 = MLP(64, 128, activation_first=True, hidden_layer=3, hidden_dim=hidden_dim, activation = 'tanh')
        #self.graph_e_3 = TAGConv(128, encoding_size, activation=nn.GELU())
        #self.graph_d_3 = TAGConv(32, 128, activation=nn.GELU())

        #self.decode_MLP = MLP(encoding_size, 3, hidden_layer=3, hidden_dim=128, activation='leaky_relu')
        self.norm = nn.LayerNorm(encoding_size)
        #self.learn_forces = MLP(encoding_size, 3, hidden_layer=3, hidden_dim=128, activation='leaky_relu')
        #self.decode_vel = MLP(encoding_size, 3, hidden_layer=3, hidden_dim=128, activation='leaky_relu')
        
        #self.layer1 = EGNN(dim = 128)

        # (1, 16, 512), (1, 16, 3)


    def calc_edge_feat(self,
                       src_idx: torch.Tensor,
                       dst_idx: torch.Tensor,
                       pos_src: torch.Tensor,
                       pos_dst=None) -> torch.Tensor:
        # this is the raw input feature

        # to enhance computation performance, dont track their calculation on graph
        if pos_dst is None:
            pos_dst = pos_src

        with torch.no_grad():
            rel_pos = pos_dst[dst_idx.long()] - pos_src[src_idx.long()]
            if isinstance(self.box_size, torch.Tensor):
                rel_pos_periodic = torch.remainder(rel_pos + 0.5 * self.box_size.to(rel_pos.device),
                                                   self.box_size.to(rel_pos.device)) - 0.5 * self.box_size.to(rel_pos.device)
            else:
                rel_pos_periodic = torch.remainder(rel_pos + 0.5 * self.box_size,
                                                   self.box_size) - 0.5 * self.box_size

            rel_pos_norm = rel_pos_periodic.norm(dim=1).view(-1, 1)  # [edge_num, 1]
            rel_pos_periodic /= rel_pos_norm + 1e-8   # normalized

        if self.training:
            self.fit_length(rel_pos_norm)
            self._update_length_stat(self.length_scaler.mean_, np.sqrt(self.length_scaler.var_))

        rel_pos_norm = (rel_pos_norm - self.length_mean) / self.length_std
        edge_feat = torch.cat((rel_pos_periodic,
                               rel_pos_norm,
                               self.edge_expand(rel_pos_norm)), dim=1)
        return edge_feat

    def build_graph(self,
                    fluid_edge_idx: torch.Tensor,
                    fluid_pos: torch.Tensor,
                    fluid_vel: torch.Tensor,
                    self_loop=True) -> dgl.DGLGraph:

        center_idx = fluid_edge_idx[0, :]  # [edge_num, 1]
        neigh_idx = fluid_edge_idx[1, :]
        fluid_graph = dgl.graph((neigh_idx, center_idx))
        
        fluid_edge_feat = self.calc_edge_feat(center_idx, neigh_idx, fluid_pos)
        fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat))  # [edge_num, 64]
        fluid_edge_emb = self.edge_encoder(fluid_edge_feat)  # [edge_num, 64]
        fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
        fluid_graph.edata['e'] = fluid_edge_emb
        


        #add node embeddings
        #fluid_graph.ndata['e'] = fluid_pos
        positions_encoded = self.node_encoder(fluid_pos)
        velocities_encoded = self.vel_encoder(fluid_vel)
        
        new_node = torch.cat((positions_encoded, velocities_encoded), axis=1)
        fluid_graph.ndata['e'] = new_node
        #fluid_graph.ndata['e'] = positions_encoded + velocities_encoded

        #graph_to_save = dgl.graph((neigh_idx, center_idx))
        #graph_to_save.ndata['e'] = self.node_encoder(fluid_pos)

        # add self loop for fluid particles
        if self_loop:
            fluid_graph.add_self_loop()
            #graph_to_save.add_self_loop()

        """
        folder_path = "graphs_to_train_embedding256"
        os.makedirs(folder_path, exist_ok=True)

        graph_filename = os.path.join(folder_path, f"graph{self.save_index}.dgl")
        self.save_index += 1
        dgl.save_graphs(graph_filename, graph_to_save)
        """

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

    def gencoder_mine(self, h: torch.Tensor, g: dgl.DGLGraph):
        out = self.graph_conv(h, g)
        out = self.norm(out)
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
        
        old_graph_nodes = fluid_graph.ndata['e']

        g_embed = self.gencoder_mine(fluid_graph.ndata['e'], fluid_graph)

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
        
    def gdecoder_MLP(self, h: torch.Tensor):

        out = self.decode_MLP(h)
        vel = self.decode_vel(h)
        return h, out, vel

    def forward(self, h:torch.Tensor):

        h, out, vel = self.gdecoder_MLP(h)
        return h, out, vel

    def load_from_checkpoint(self, checkpoint_path):
        # Your loading logic here
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)
        return self
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

class f1(nn.Module):
    def __init__(self, encoding_size):
        super().__init__()
        embed1 = 128
        embed2 = 256
        self.encoding_size = encoding_size
        self.func0 = MLP(encoding_size, embed1, hidden_layer=3, hidden_dim=128, activation='softplus')
        self.func1 = MLP(embed1, embed1, hidden_layer=4, hidden_dim=256, activation='softplus')
        self.func2 = MLP(embed1, embed2, hidden_layer=4, hidden_dim=512, activation='softplus')
        self.func3 = MLP(embed2, encoding_size, hidden_layer=4, hidden_dim=256, activation='softplus')
        #nn.init.normal_(self.dense14.weight.data, 0.0, 0.0)
        #nn.init.normal_(self.dense14.bias.data, 0.0, 0.0)
          
    def forward(self, t, x):
        out = self.func0(x)
        out = self.func1(out)
        out = self.func2(out)
        out = self.func3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        #self.ode_lin1 = MLP(odefunc.encoding_size, odefunc.encoding_size, hidden_layer=1, hidden_dim=128, activation='softplus')
        #self.ode_lin2 = MLP(odefunc.encoding_size, odefunc.encoding_size, hidden_layer=1, hidden_dim=128, activation='softplus')
        

    def forward(self, x, integration_time):

        #out = self.ode_lin1(x)
        out = odeint(self.odefunc, x, integration_time, atol=1e-6, rtol=1e-6)
        #out = self.ode_lin2(out)

        #out = torchdiffeq.autograd.odeint(self.odefunc, x, integration_time, method='rk4', options={'atol': 1e-6, 'rtol': 1e-6})

        return out

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity='tanh',
        )

        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        output, hn = self.rnn(x)
        output = self.linear(output)

        return output   

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, proj_size):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            proj_size=proj_size,
            nonlinearity='tanh',
        )

    def forward(self, x):
        output, hn = self.rnn(x)

        return output

class Args_latentODE:
    def __init__(self):
        self.latents = 32 ## Size of the latent state
        self.poisson = True ## Model poisson-process likelihood for the density of events in addition to reconstruction
        self.units = 100 ## Number of units per layer in ODE func
        self.gen_layers = 2 ## Number of layers in ODE func in generative ODE
        self.rec_dims = 20 ## Dimensionality of the recognition model (ODE or RNN)
        self.z0_encoder = 'odernn' ## Type of encoder for Latent ODE model: odernn or rnn
        self.rec_layers = 1 ## Number of layers in ODE func in recognition ODE
        self.gru_units = 200 ## Number of units per layer in each of GRU update networks

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

        self.latentODE = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device)
        self.mode = mode

    def forward(self, truth, t):
        if self.mode == 'train': ## try for exrapolation: We encode the observations in the first half forward in time and reconstruct the second half.
            ## interpolation training
            mask, masked_indices, not_masked_indices = create_mask(truth, 0.3)
            truth = mask*truth
            time_steps_to_predict = t
            truth_time_steps = t[not_masked_indices]
            """
            ## extrapolation training
            time_steps_to_predict = torch.arange(0,t.size()[0]).to("cuda").float()
            truth_time_steps = torch.arange(0,int(t.size()[0]/2)).to("cuda").float()
            ones_tensor = torch.ones(truth.size()).to("cuda")
            zeros_tensor = torch.zeros(truth.size()).to("cuda")
            truth = torch.cat((truth, zeros_tensor))
            mask = torch.cat((ones_tensor, zeros_tensor))
            """

        if self.mode == 'test':
            time_steps_to_predict = torch.arange(0,2*t.size()[0]).to("cuda").float()
            truth_time_steps = torch.arange(0,t.size()[0]).to("cuda").float()
            ones_tensor = torch.ones(truth.size()).to("cuda")
            zeros_tensor = torch.zeros(truth.size()).to("cuda")
            truth = torch.cat((truth, zeros_tensor))
            mask = torch.cat((ones_tensor, zeros_tensor))

        output, extra_info = self.latentODE.get_reconstruction(time_steps_to_predict,
                                            truth.permute(1, 0, 2),
                                            truth_time_steps,
                                            mask = mask.permute(1, 0, 2),
                                            n_traj_samples = 1,
                                            run_backwards = True,
                                            mode = None)

        return output


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

        PATH1 = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/AUTOENCODER/encoder_checkpoint_29.ckpt'
        PATH2 = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/AUTOENCODER/decoder_checkpoint_29.ckpt'

        self.encoding_size = encoding_size

        self.encoder = Encoder(encoding_size,
                    out_feats,
                    box_size,   # can also be array
                    hidden_dim=128,
                    conv_layer=2,
                    edge_embedding_dim=32,
                    dropout=0.1,
                    drop_edge=True,
                    use_layer_norm=False).load_from_checkpoint(PATH1)

        self.encoder.freeze()

        self.architecture = architecture
        
        if architecture == 'node':
            self.neuralODE = ODEBlock(f1(encoding_size))

        if architecture == 'latentode':
            self.latentODE = latentODE(encoding_size, mode)

        if architecture == 'recurrent':
            self.recurrent = RNN(input_size = encoding_size, hidden_size = 256, num_layers=8)
            #self.recurrent = LSTM(input_size = encoding_size, hidden_size = 256, num_layers=8, proj_size = encoding_size)

        self.decoder = Decoder(
                    encoding_size,
                    out_feats,
                    box_size,   # can also be array
                    hidden_dim=128,
                    use_layer_norm=False).load_from_checkpoint(PATH2)
            
        self.decoder.freeze()

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                fluid_edge_lst: List[torch.Tensor],
                fluid_vel_lst: List[torch.Tensor],
                t,
                ):

        encoded = self.encoder(fluid_pos_lst, fluid_edge_lst, fluid_vel_lst).view(len(fluid_pos_lst), 258, self.encoding_size)

        if self.architecture == 'node':
            to_decode = self.neuralODE(encoded, t)
        
        if self.architecture == 'latentode':
            to_decode = self.latentODE(encoded, t).squeeze(0).permute(1,0,2)
        
        if self.architecture == 'recurrent':
            to_decode = self.recurrent(encoded)

        h, pos_result, vel_result = self.decoder(to_decode)

        return pos_result, vel_result, h

class Autoencoder(nn.Module):
    def __init__(self,
                encoding_size,
                out_feats,
                box_size,   # can also be array
                hidden_dim=128,
                conv_layer=2,
                edge_embedding_dim=32,
                dropout=0.1,
                drop_edge=True,
                use_layer_norm=False,):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(encoding_size,
                    out_feats,
                    box_size,   # can also be array
                    hidden_dim=128,
                    conv_layer=2,
                    edge_embedding_dim=32,
                    dropout=0.1,
                    drop_edge=True,
                    use_layer_norm=False)

        self.decoder = Decoder(
                    encoding_size,
                    out_feats,
                    box_size,   # can also be array
                    hidden_dim=128,
                    use_layer_norm=False)

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                fluid_edge_lst: List[torch.Tensor],
                fluid_vel_lst: List[torch.Tensor],
                ):

        embed = self.encoder(fluid_pos_lst, fluid_edge_lst, fluid_vel_lst)

        h, pos_result, vel_result = self.decoder(embed)

        return pos_result, vel_result

## TO DO: fix RNN

class RNN_entireModel(nn.Module):
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
        super(EntireModel, self).__init__()

        PATH1 = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/AUTOENCODER/encoder_checkpoint_29.ckpt'
        PATH2 = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/AUTOENCODER/decoder_checkpoint_29.ckpt'

        self.encoder = Encoder(encoding_size,
                    out_feats,
                    box_size,   # can also be array
                    hidden_dim=128,
                    conv_layer=2,
                    edge_embedding_dim=32,
                    dropout=0.1,
                    drop_edge=True,
                    use_layer_norm=False).load_from_checkpoint(PATH1)

        self.encoder.freeze()
        
        self.rnn = RNN(encoding_size, hidden_dim = 128, num_layers = 5)

        self.decoder = Decoder(
                    encoding_size,
                    out_feats,
                    box_size,   # can also be array
                    hidden_dim=128,
                    use_layer_norm=False).load_from_checkpoint(PATH2)
        self.decoder.freeze()

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                fluid_edge_lst: List[torch.Tensor],
                fluid_vel_lst: List[torch.Tensor],
                ):

        encode_first = self.encoder(fluid_pos_lst, fluid_edge_lst, fluid_vel_lst)

        to_decode = self.rnn(encode_first)

        h, pos_result, vel_result = self.decoder(to_decode)

        return pos_result, vel_result