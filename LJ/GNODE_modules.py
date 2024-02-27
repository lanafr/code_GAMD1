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

import dgl.data
import matplotlib.pyplot as plt
import networkx as nx

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

#import torchcde

import wandb

import torch.utils.data as data

from LATENT_ODE.create_latent_ode_model import *
from torch.distributions.normal import Normal
import LATENT_ODE.utils as utils

from graph_utils import NeighborSearcher, graph_network_nbr_fn

BOX_SIZE = 27.27
CUTOFF_RADIUS = 7.5
NUM_OF_ATOMS = 258

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
                 hidden_dim=64,
                 conv_layer=2,  # at some point try with 3
                 edge_emb_dim=16,
                 use_layer_norm=False,
                 use_batch_norm=False, ##changed it
                 drop_edge=False,
                 activation='relu',
                 update_egde_emb=False,
                 box_size = BOX_SIZE,
                 ):
        super(SmoothConvBlockNew, self).__init__()
        
        self.edge_emb_dim = edge_emb_dim
        self.edge_expand = RBFExpansion(high=1, gap=0.025)
        self.edge_drop_out = nn.Dropout(0.1)

        self.length_mean = nn.Parameter(torch.tensor([0.]), requires_grad=False)
        self.length_std = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.length_scaler = StandardScaler()

        self.edge_encoder = MLP(6 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
                                activation='softplus') # changed ittt
        self.edge_layer_norm = nn.LayerNorm(self.edge_emb_dim)
        #self.graph_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')
        self.save_index = 0

        self.decoder = MLP(16, 6, hidden_dim = hidden_dim, hidden_layer=2)

        #self.decode_MLP = MLP(encoding_size, 3, hidden_layer=3, hidden_dim=128, activation='leaky_relu')
        self.norm = nn.LayerNorm(16)#in_node_feats)

        if isinstance(box_size, np.ndarray):
            self.box_size = torch.from_numpy(box_size).float()
        else:
            self.box_size = box_size

        self.conv = nn.ModuleList()
        self.edge_emb_dim = edge_emb_dim
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm

        self.drop_edge = drop_edge
        self.build_graph_from_pos=build_graph_from_pos()
        """
        self.make_a_graph = make_a_graph(in_node_feats,
                    27.27,   # can also be array
                    hidden_dim=128,
                    conv_layer=3,
                    edge_embedding_dim=32,
                    dropout=0.1,
                    drop_edge=True,
                    use_layer_norm=False)
        """
        if use_batch_norm == use_layer_norm and use_batch_norm:
            raise Exception('Only one type of normalization at a time')
        if use_layer_norm or use_batch_norm:
            self.norm_layers = nn.ModuleList()

        self.node_embed = MLP(6, 16, hidden_dim=hidden_dim, hidden_layer=2)

        for layer in range(conv_layer):
            if layer == 0:
                self.conv.append(SmoothConvLayerNew(in_node_feats=16,#in_node_feats,
                                                 in_edge_feats=self.edge_emb_dim,
                                                 out_node_feats=16,#out_node_feats,
                                                 hidden_dim=hidden_dim,
                                                 activation=activation,
                                                 drop_edge=drop_edge,
                                                 update_edge_emb=update_egde_emb))
            else:
                self.conv.append(SmoothConvLayerNew(in_node_feats=16,#in_node_feats,
                                                 in_edge_feats=self.edge_emb_dim,
                                                 out_node_feats=16,#out_node_feats,
                                                 hidden_dim=hidden_dim,
                                                 activation=activation,
                                                 drop_edge=drop_edge,
                                                 update_edge_emb=update_egde_emb))
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(16))#out_node_feats))
            elif use_batch_norm:
                self.norm_layers.append(nn.BatchNorm1d(16))#out_node_feats)) 

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

    def _update_length_stat(self, new_mean, new_std):
        self.length_mean[0] = new_mean[0]
        self.length_std[0] = new_std[0]

    def fit_length(self, length):
        if not isinstance(length, np.ndarray):
            length = length.detach().cpu().numpy().reshape(-1, 1)
        self.length_scaler.partial_fit(length)

    def forward(self, cat) -> torch.Tensor:
        ## build a graph
        g = self.build_graph_from_pos.get_it(cat)
        ## simple node embedding
        h = self.node_embed(g.ndata['e'])

        ## odge embedding from GAMD
        edges = g.edges()
        # Extract the center indices and neighbor indices
        center_idx = edges[1]  # This corresponds to the first node ID in each edge
        neigh_idx = edges[0]
        fluid_edge_feat = self.calc_edge_feat(center_idx, neigh_idx, cat)
        fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat))  # [edge_num, 64]
        fluid_edge_emb = self.edge_encoder(fluid_edge_feat)  # [edge_num, 64]
        fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
        g.edata['e'] = fluid_edge_emb

        g.add_self_loop()

        ## conv layer through which graph neural ode will
        for l, conv_layer in enumerate(self.conv):
            if self.use_layer_norm or self.use_batch_norm:
                h = conv_layer.forward(g, self.norm_layers[l](h)) + h
            else:
                h = conv_layer.forward(g, h) + h

        h = self.decoder(h)

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


class GDEFunc(nn.Module):
    def __init__(self, gnn:nn.Module):
        """General GDE function class. To be passed to an ODEBlock"""
        super().__init__()
        self.gnn = gnn
        self.nfe = 0
    
    def set_graph(self, g:dgl.DGLGraph):
        for layer in self.gnn:
            layer.g = g
            
    def forward(self, t, x):
        self.nfe += 1
        x = self.gnn(x)
        return x
    
class ControlledGDEFunc(GDEFunc):
    def __init__(self, gnn:nn.Module):
        """ Controlled GDE version. Input information is preserved longer via hooks to input node features X_0, 
            affecting all ODE function steps. Requires assignment of '.h0' before calling .forward"""
        super().__init__(gnn)
        self.nfe = 0
            
    def forward(self, t, x):
        self.nfe += 1
        x = torch.cat([x, self.h0], 1)
        x = self.gnn(x)
        return x

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        #self.ode_lin1 = MLP(odefunc.encoding_size, odefunc.encoding_size, hidden_layer=1, hidden_dim=128, activation='softplus')
        #self.ode_lin2 = MLP(odefunc.encoding_size, odefunc.encoding_size, hidden_layer=1, hidden_dim=128, activation='softplus')
        

    def forward(self, integration_time, x):
        #out = self.ode_lin1(x)
        out = odeint(self.odefunc, x, integration_time, atol=1e-3, rtol=1e-3, method ='rk4')
        #out = self.ode_lin2(out)

        #out = torchdiffeq.autograd.odeint(self.odefunc, x, integration_time, method='rk4', options={'atol': 1e-6, 'rtol': 1e-6})

        return out

class EntireModel(nn.Module):
    def __init__(self,
                encoding_size,
                out_feats,
                box_size,   # can also be array
                mode,
                architecture,
                hidden_dim=128,
                conv_layer=3,
                edge_embedding_dim=32,
                dropout=0.1,
                drop_edge=True,
                use_layer_norm=False):
        super(EntireModel, self).__init__()

        self.encoding_size = encoding_size

        self.GNODE = ODEBlock(GDEFunc(SmoothConvBlockNew(in_node_feats=encoding_size,
                                             out_node_feats=encoding_size,
                                             hidden_dim=hidden_dim,
                                             conv_layer=conv_layer,
                                             edge_emb_dim=edge_embedding_dim,
                                             use_layer_norm=use_layer_norm,
                                             use_batch_norm=not use_layer_norm,
                                             drop_edge=drop_edge,
                                             activation='softplus')))

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],
                fluid_vel_lst: List[torch.Tensor],
                t,
                ):

        inp = torch.cat((fluid_pos_lst[0], fluid_vel_lst[0]), dim=1)

        result = self.GNODE(t, inp)
        pos_result = result[:,:, :3]
        vel_result = result[:,:, 3:]
        graph_emb = torch.zeros_like(pos_result)

        #h, pos_result, vel_result, forces = self.decoder(graph_emb)

        return pos_result, vel_result, graph_emb


class build_graph_from_pos:
    def __init__(self,):
        super(build_graph_from_pos, self).__init__()
        self.nbr_cache = {}
        self.cutoff = CUTOFF_RADIUS
        self.nbr_searcher = NeighborSearcher(BOX_SIZE, self.cutoff)
        self.nbrlst_to_edge_mask = jax.jit(graph_network_nbr_fn(self.nbr_searcher.displacement_fn,
                                                                    self.cutoff,
                                                                    NUM_OF_ATOMS))
    
    def get_edge_idx(self, nbrs, pos_jax, mask):
        dummy_center_idx = nbrs.idx.copy()
        #dummy_center_idx = jax.ops.index_update(dummy_center_idx, None,
        #                                        jnp.arange(pos_jax.shape[0]).reshape(-1, 1))
        dummy_center_idx = dummy_center_idx.at[None].set(jnp.arange(pos_jax.shape[0]).reshape(-1, 1))
        center_idx = dummy_center_idx.reshape(-1)
        center_idx_ = cupy.asarray(center_idx)
        center_idx_tsr = torch.as_tensor(center_idx_, device='cuda')

        neigh_idx = nbrs.idx.reshape(-1)

        # cast jax device array to cupy array so that it can be transferred to torch
        neigh_idx = cupy.asarray(neigh_idx)
        mask = cupy.asarray(mask)
        mask = torch.as_tensor(mask, device='cuda')
        flat_mask = mask.view(-1)
        neigh_idx_tsr = torch.as_tensor(neigh_idx, device='cuda')

        edge_idx_tsr = torch.cat((center_idx_tsr[flat_mask].view(1, -1), neigh_idx_tsr[flat_mask].view(1, -1)),
                                 dim=0)
        return edge_idx_tsr

    def search_for_neighbor(self, pos, nbr_searcher, masking_fn, type_name):
        pos_jax = jax.device_put(pos, jax.devices("gpu")[0])

        if not nbr_searcher.has_been_init:
            nbrs = nbr_searcher.init_new_neighbor_lst(pos_jax)
            self.nbr_cache[type_name] = nbrs
        else:
            nbrs = nbr_searcher.update_neighbor_lst(pos_jax, self.nbr_cache[type_name])
            self.nbr_cache[type_name] = nbrs

        edge_mask_all = masking_fn(pos_jax, nbrs.idx)
        edge_idx_tsr = self.get_edge_idx(nbrs, pos_jax, edge_mask_all)
        return edge_idx_tsr.long()

    def get_it(self, cat):

        pos_np = cat[:, :3].detach().cpu().numpy()
        edge_idx_tsr = self.search_for_neighbor(pos_np,
                                                    self.nbr_searcher,
                                                    self.nbrlst_to_edge_mask,
                                                    'all')

        center_idx = edge_idx_tsr[0, :]  # [edge_num, 1]
        neigh_idx = edge_idx_tsr[1, :]
        graph_now = dgl.graph((neigh_idx, center_idx))

        graph_now.ndata['e']=cat

        return graph_now