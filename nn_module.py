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

from torchdyn.core import NeuralODE
from torchdyn import *  
from torchdyn.models import *
from torchdyn.nn import *


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
"""
class NeuralOrdinaryDE(nn.Module):
    def __init__(self,
                f,
                sensitivity,
                solver):
        super(NeuralOrdinaryDE, self).__init__()
        #self.model = NeuralODE(f, sensitivity=sensitivity, solver=solver)
        self.model = NeuralODE (f, sensitivity='interpolated_adjoint', solver='tsit5', atol=1e-5, rtol=1e-5)
    
    def forward(self, x):
        return self.model(x)
        
"""
class SmoothConvLayerNew(nn.Module):
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 hidden_dim=128,
                 activation='relu',
                 #drop_edge=True,
                 drop_edge=False,
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
                 edge_emb_dim=64,
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
                h = conv_layer.forward(graph, self.norm_layers[l](h)) + h ## I changed ittttt
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


class SimpleMDNetNew(nn.Module):  # no bond, no learnable node encoder
    def __init__(self,
                 encoding_size,
                 out_feats,
                 box_size,   # can also be array
                 hidden_dim=128,
                 conv_layer=4,
                 edge_embedding_dim=256,
                 dropout=0.1,
                 drop_edge=True,
                 use_layer_norm=False):
        super(SimpleMDNetNew, self).__init__()
        """
        self.graph_conv = SmoothConvBlockNew(in_node_feats=encoding_size,
                                             out_node_feats=encoding_size,
                                             hidden_dim=hidden_dim,
                                             conv_layer=conv_layer,
                                             edge_emb_dim=edge_embedding_dim,
                                             use_layer_norm=use_layer_norm,
                                             use_batch_norm=not use_layer_norm,
                                             drop_edge=drop_edge,
                                             activation='silu')
        """

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

        self.node_encoder = MLP(3, encoding_size, hidden_layer=4, hidden_dim=hidden_dim, activation='leaky_relu')
        self.node_dencoder = MLP(encoding_size, 3, hidden_layer=4, hidden_dim=hidden_dim, activation='leaky_relu')
        #self.edge_encoder = MLP(3 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
        #                        activation='gelu')
        #self.edge_layer_norm = nn.LayerNorm(self.edge_emb_dim)
        #self.graph_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')
        self.save_index = 0


        #embedding_size
        
        self.graph_conv1 = GraphConv(encoding_size, hidden_dim, norm='both', weight=True, bias=True)#, activation = nn.SiLU())
        self.graph_conv2 = GraphConv(hidden_dim, encoding_size, norm='both', weight=True, bias=True)#, activation = nn.Sigmoid())
        self.graph_conv_hid1 = GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True, activation = nn.Tanh())
        self.graph_conv_hid2 = GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True, activation = nn.Tanh())
        #self.graphMLP1 = MLP(128, 64, activation_first=True, hidden_layer=3, hidden_dim=hidden_dim, activation = 'tanh')
        #self.graphMLP2 = MLP(64, 128, activation_first=True, hidden_layer=3, hidden_dim=hidden_dim, activation = 'tanh')
        self.graph_conv1_hid1 = GraphConv(128, 128, norm='both', weight=True, bias=True, activation = nn.SiLU())
        self.graph_conv2_hid2 = GraphConv(128, 128, norm='both', weight=True, bias=True, activation = nn.SiLU())
        


        """

        self.graph_conv1 = GraphConv(3, 256, norm='both', weight=True, bias=True)#, activation = nn.SiLU())
        self.graph_conv2 = GraphConv(256, 3, norm='both', weight=True, bias=True)#, activation = nn.Sigmoid())
        self.graph_conv_hid1 = GraphConv(256, hidden_dim, norm='both', weight=True, bias=True, activation = nn.SiLU())
        self.graph_conv_hid2 = GraphConv(hidden_dim, 256, norm='both', weight=True, bias=True, activation = nn.SiLU())
        self.graphMLP1 = MLP(hidden_dim, 64, activation_first=True, hidden_layer=10, hidden_dim=hidden_dim, activation = 'silu')
        self.graphMLP2 = MLP(64, hidden_dim, activation_first=True, hidden_layer=10, hidden_dim=hidden_dim, activation = 'silu')
        """


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
                    self_loop=True) -> dgl.DGLGraph:

        center_idx = fluid_edge_idx[0, :]  # [edge_num, 1]
        neigh_idx = fluid_edge_idx[1, :]
        fluid_graph = dgl.graph((neigh_idx, center_idx))
        
        #fluid_edge_feat = self.calc_edge_feat(center_idx, neigh_idx, fluid_pos)

        #fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat))  # [edge_num, 64]
        #fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
        #fluid_graph.edata['e'] = fluid_edge_emb
        


        #add node embeddings
        #fluid_graph.ndata['e'] = fluid_pos
        fluid_graph.ndata['e'] = self.node_encoder(fluid_pos)

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

    def build_graph_batches(self, pos_lst, edge_idx_lst):
        graph_lst = []
        for pos, edge_idx in zip(pos_lst, edge_idx_lst):
            graph = self.build_graph(edge_idx, pos)
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
        x = self.graph_conv1(g,h)
        x = self.graph_conv_hid1(g,x)
        x = self.graph_conv1_hid1(g,x)
        #x = self.graphMLP1(x)
        return x
     

    def gdecoder_mine(self, h: torch.Tensor, g: dgl.DGLGraph):
        #x = self.graphMLP2(h)

        x = self.graph_conv2_hid2(g,h)
        x = self.graph_conv_hid2(g,x)
        x = self.graph_conv2(g,x)
        return x

    

    def forward(self,
                fluid_pos_lst: List[torch.Tensor],  # list of [N, 3]
                fluid_edge_lst: List[torch.Tensor]
                ) -> torch.Tensor:
        if len(fluid_pos_lst) > 1:
            fluid_graph = self.build_graph_batches(fluid_pos_lst, fluid_edge_lst)
        else:
            fluid_graph = self.build_graph(fluid_edge_lst[0], fluid_pos_lst[0])
        
        old_graph_nodes = fluid_graph.ndata['e']

        g_embed = self.gencoder_mine(fluid_graph.ndata['e'], fluid_graph)
        fluid_graph.ndata['e'] = self.gdecoder_mine(g_embed, fluid_graph)
        pos = self.node_dencoder(fluid_graph.ndata['e'])

        return pos, old_graph_nodes, fluid_graph.ndata['e'], g_embed
        #return pos


class GCNLayer_for_autoencoding(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer_for_autoencoding, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.

        gcn_msg = fn.copy_u(u="e", out="m")
        gcn_reduce = fn.sum(msg="m", out="e")

        with g.local_scope():
            g.ndata["e"] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata["e"]
            return self.linear(h)

class GNNAutoencoder(nn.Module):
    def __init__(self,
                 encoding_size,
                 out_feats,
                 box_size,   # can also be array
                 hidden_dim=128,
                 conv_layer=2,
                 edge_embedding_dim=1,
                 dropout=0.1,
                 drop_edge=True,
                 use_layer_norm=False):
        super(GNNAutoencoder, self).__init__()
        """
        self.graph_conv = SmoothConvBlockNew(in_node_feats=encoding_size,
                                             out_node_feats=64,
                                             hidden_dim=hidden_dim,
                                             conv_layer=conv_layer,
                                             edge_emb_dim=1,
                                             use_layer_norm=use_layer_norm,
                                             use_batch_norm=not use_layer_norm,
                                             drop_edge=drop_edge,
                                             activation='silu')
        """

        
        self.graph_conv1 = GraphConv(encoding_size, hidden_dim, norm='both', weight=True, bias=True)#, activation = nn.SiLU())
        #self.graph_conv1 = MLP(encoding_size, hidden_dim, activation_first=True, hidden_layer=1, hidden_dim=hidden_dim, activation = 'silu')
        self.graph_conv2 = GraphConv(hidden_dim, encoding_size, norm='both', weight=True, bias=True)#, activation = nn.Sigmoid())
        #self.graph_conv2 = MLP(hidden_dim, encoding_size, activation_first=True, hidden_layer=1, hidden_dim=hidden_dim, activation = 'silu')
        #self.graph_conv1_hid1 = GraphConv(128, 64, norm='both', weight=True, bias=True, activation = nn.SiLU())
        #self.graph_conv2_hid2 = GraphConv(64, 128, norm='both', weight=True, bias=True, activation = nn.SiLU())
        self.graph_conv_hid1 = GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True, activation = nn.SiLU())
        self.graph_conv_hid2 = GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True, activation = nn.SiLU())
        self.graphMLP1 = MLP(hidden_dim, 64, activation_first=True, hidden_layer=1, hidden_dim=hidden_dim, activation = 'silu')
        self.graphMLP2 = MLP(64, hidden_dim, activation_first=True, hidden_layer=1, hidden_dim=hidden_dim, activation = 'silu')
        
        
        """
        self.graph_conv1 = GCNLayer_for_autoencoding(encoding_size, hidden_dim)
        self.graph_conv2 = GCNLayer_for_autoencoding(hidden_dim, encoding_size)
        self.graph_conv1_hid1 = GCNLayer_for_autoencoding(hidden_dim, 64)
        self.graph_conv2_hid2 = GCNLayer_for_autoencoding(64, hidden_dim)
        self.graph_conv_hid1 = GCNLayer_for_autoencoding(hidden_dim, hidden_dim)
        self.graph_conv_hid2 = GCNLayer_for_autoencoding(hidden_dim, hidden_dim)
        """

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

        #self.node_encoder = MLP(3, encoding_size, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')
        #self.edge_encoder = MLP(3 + 1 + len(self.edge_expand.centers), self.edge_emb_dim, hidden_dim=hidden_dim,
                                #activation='gelu')
        #self.edge_layer_norm = nn.LayerNorm(self.edge_emb_dim)
        #self.graph_decoder = MLP(encoding_size, out_feats, hidden_layer=2, hidden_dim=hidden_dim, activation='gelu')

        """
        layers_en = [GraphConv(encoding_size, hidden_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(hidden_dim, 64, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(hidden_dim, 64, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers_en = nn.ModuleList(layers_en)

        layers_de = [GraphConv(64, 128, activation=F.relu, allow_zero_in_degree=True),
                 GraphConv(128, hidden_dim, activation=lambda x: x, allow_zero_in_degree=True),
                 GraphConv(hidden_dim, encoding_size, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers_de = nn.ModuleList(layers_de)
        """

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

    """
    def build_graph(self,
                    fluid_edge_idx: torch.Tensor,
                    fluid_pos: torch.Tensor,
                    self_loop=True) -> dgl.DGLGraph:

        center_idx = fluid_edge_idx[0, :]  # [edge_num, 1]
        neigh_idx = fluid_edge_idx[1, :]
        fluid_graph = dgl.graph((neigh_idx, center_idx))
        #fluid_edge_feat = self.calc_edge_feat(center_idx, neigh_idx, fluid_pos)
        
        #fluid_edge_emb = self.edge_layer_norm(self.edge_encoder(fluid_edge_feat))  # [edge_num, 64]
        #fluid_edge_emb = self.edge_drop_out(fluid_edge_emb)
        #fluid_graph.edata['e'] = fluid_edge_emb
        
        #fluid_graph.edata['e'] = 1 ## maybe we don't need edge embeddings

        #add node embeddings
        fluid_graph.ndata['e'] = self.node_encoder(fluid_pos)

        # add self loop for fluid particles
        if self_loop:
            fluid_graph.add_self_loop()
        return fluid_graph

    def build_graph_batches(self, pos_lst, edge_idx_lst):
        graph_lst = []
        for pos, edge_idx in zip(pos_lst, edge_idx_lst):
            graph = self.build_graph(edge_idx, pos)
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
    """
    
    def gencoder_mine(self, h: torch.Tensor, g: dgl.DGLGraph):
        x = self.graph_conv1(g,h)
        x = self.graph_conv_hid1(g,x)
        #x = self.graph_conv1_hid1(g,x)
        x = self.graphMLP1(x)
        return x
     

    def gdecoder_mine(self, h: torch.Tensor, g: dgl.DGLGraph):
        x = self.graphMLP2(h)
        #x = self.graph_conv2_hid2(g,h)
        x = self.graph_conv_hid2(g,x)
        x = self.graph_conv2(g,x)
        return x

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        
        g_embed = self.gencoder_mine(g.ndata['e'], g)
        g.ndata['e'] = self.gdecoder_mine(g_embed, g)

        return g