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
model1 = ParticleNetLightning(args).load_from_checkpoint(PATH, args=args)
model1.load_training_stats(SCALER_CKPT)
model1.cuda()
model1.eval()

PATH = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/sequential_network/checkpoint_29.ckpt'
SCALER_CKPT = '/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/sequential_network/scaler_29.npz'
args = SimpleNamespace(use_layer_norm=False,
                    encoding_size=64,
                    hidden_dim=128,
                    drop_edge=False,
                    conv_layer=4,
                    rotate_aug=False,
                    update_edge=False,
                    use_part=False,
                    data_dir='',
                    loss='mse')
model2 = ParticleNetLightning(args).load_from_checkpoint(PATH, args=args)
model2.load_training_stats(SCALER_CKPT)
model2.cuda()
model2.eval()


## which one to start with
start_all = np.load(f'md_dataset/lj_data/data_3_567.npz')
start_pos = start_all['pos']
pos_hopefully_same, graph1, graph2, embed = model1.predict_nextpos(start_pos)

t = 50

next_embeddings = model2.ode_embed_func(embed,t)

trajectory = model2.decode_the_sequence(next_embeddings, graph1, t)

trajectory = prepend(start_pos, trajectory)







