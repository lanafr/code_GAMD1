import sys, os


import argparse
import os, sys
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import jax
import jax.numpy as jnp
import cupy
from pytorch_lightning.loggers import WandbLogger
import dgl.nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from graph_utils import NeighborSearcher, graph_network_nbr_fn
import time
sys.path.append(os.path.join('../',os.path.dirname(os.path.abspath(''))))
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
sys.path.append('/home/guests/lana_frkin/GAMDplus/code')
sys.path.append('/home/guests/lana_frkin/GAMDplus/code/LJ')
print(sys.path)

from nn_module import GNNAutoencoder
from train_utils import Graphs_data
from train_autoencoder import AutoencoderNetLightning, NUM_OF_ATOMS
import torch
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch.nn as nn

"""
dataset = Graphs_data(dataset_path='/home/guests/lana_frkin/GAMDplus/code/LJ/graphs_to_train',
                               sample_num=9055, #this I changed
                               case_prefix='graphs_to_train',
                               mode='train')

distributed_sampler = DistributedSampler(dataset, seed=0)
dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=4, shuffle=False, drop_last=True, num_workers=2, sampler=distributed_sampler)
"""

PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt_autoencoder/autoencoder_for_graphs1/checkpoint_29.ckpt'
SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt_autoencoder/autoencoder_for_graphs1/scaler_29.npz'
args = SimpleNamespace(use_layer_norm=False,
                       encoding_size=128,
                       hidden_dim=256,
                       edge_embedding_dim=1,
                       drop_edge=False,
                       conv_layer=4,
                       rotate_aug=False,
                       update_edge=False,
                       use_part=False,
                       data_dir='',
                       loss='mae')
model = AutoencoderNetLightning(args).load_from_checkpoint(PATH, args=args)
model.load_training_stats(SCALER_CKPT)
model.cuda()
model.eval()

dataset_path='/home/guests/lana_frkin/GAMDplus/code/LJ/graphs_to_train'

idx = 3498
fname = f'graph{idx}.dgl'
data_path = os.path.join(dataset_path, fname)
graph = dgl.load_graphs(data_path)

graph_new = model.do_the_autoencoding(graph[0][0])

print("graph_old:")
print(graph[0][0].ndata['e'])
print("graph_new:")
print(graph_new.ndata['e'])
print("Loss is:")
print (nn.L1Loss()(graph_new.ndata['e'], graph[0][0].ndata['e']))


"""

#this part is to check how well the autoencoder works on a test set
pos_lst=[]
gt_lst=[]

for i in range(999):

    gt_all = np.load(f'md_dataset/lj_data_to_test/data_0_{i}.npz')
    gt = gt_all['pos']
    gt_lst.append(gt)

    pos_hopefully_same = model.predict_nextpos(gt)
    pos_lst.append(pos_hopefully_same)

gt_lst = [torch.from_numpy(arr) for arr in gt_lst]
gt_cat = torch.cat(gt_lst, dim=0)
pos_lst = [torch.from_numpy(arr) for arr in pos_lst]
pos_cat = torch.cat(pos_lst, dim=0)

mae = nn.L1Loss()(pos_cat, gt_cat)

print("Loss is:")
print(mae)

"""