###########################
# A large part of this code was taken from GAMD (https://arxiv.org/abs/2112.03383, Li, Zijie and Meidani, Kazem and Yadav, Prakarsh and Barati Farimani, Amir, 2022.)
###########################

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

import dgl.nn
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from nn_module import SimpleMDNetNew
from train_utils import LJDataNew
from graph_utils import NeighborSearcher, graph_network_nbr_fn
import time
from AGAIN_NEW_MODULESS import * ##change thiss
from train_utils_seq import *
import pint

network_has_been_trained = False


# os.environ["CUDA_VISIBLE_DEVICES"] = "" # just to test if it works w/o gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# for water box
CUTOFF_RADIUS = 7.5
BOX_SIZE = 27.27

NUM_OF_ATOMS = 258

# NUM_OF_ATOMS = 251 * 3  # tip4p
# CUTOFF_RADIUS = 3.4

LAMBDA1 = 100.
LAMBDA2 = 1e-4


def build_model(args, ckpt=None):

    param_dict = {
                  'encoding_size': args.encoding_size,
                  'out_feats': 3,
                  'hidden_dim': args.hidden_dim,
                  'edge_embedding_dim': args.edge_embedding_dim,
                  'conv_layer': 4,
                  'drop_edge': args.drop_edge,
                  'use_layer_norm': args.use_layer_norm,
                  'box_size': BOX_SIZE,
                  'mode': args.mode,
                  'architecture': args.architecture
                  }

    print("Using following set of hyper-parameters")
    print(param_dict)
    model = EntireModel(**param_dict)

    if ckpt is not None:
        print('Loading model weights from: ', ckpt)
        model.load_state_dict((torch.load(ckpt)))
    return model


class MDSimNet(pl.LightningModule):
    def __init__(self, args, num_device=1, epoch_num=100, batch_size=1, learning_rate=3e-4, log_freq=1000,
                 model_weights_ckpt=None, scaler_ckpt=None):
        super(MDSimNet, self).__init__()
        self.pnet_model = build_model(args, model_weights_ckpt)
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_device = num_device
        self.log_freq = log_freq
        self.train_data_scaler_pos = StandardScaler()
        self.train_data_scaler_vel = StandardScaler()
        self.training_mean_vel = np.array([0.])
        self.training_var_vel = np.array([1.])
        self.training_mean_pos = np.array([0.])
        self.training_var_pos = np.array([1.])
        self.mode = args.mode
        self.architecture = args.architecture

        if scaler_ckpt is not None:
            self.load_training_stats(scaler_ckpt)

        self.cutoff = CUTOFF_RADIUS
        self.nbr_searcher = NeighborSearcher(BOX_SIZE, self.cutoff)
        self.nbrlst_to_edge_mask = jax.jit(graph_network_nbr_fn(self.nbr_searcher.displacement_fn,
                                                                    self.cutoff,
                                                                    NUM_OF_ATOMS))
        self.nbr_cache = {}
        self.rotate_aug = args.rotate_aug
        self.data_dir = args.data_dir
        self.loss_fn = args.loss
        assert self.loss_fn in ['mae', 'mse']

    def load_training_stats(self, scaler_ckpt):
        if scaler_ckpt is not None:
            scaler_info = np.load(scaler_ckpt)
            self.training_mean_vel = scaler_info['mean_vel']
            self.training_var_vel = scaler_info['var_vel']
            self.training_mean_pos = scaler_info['mean_pos']
            self.training_var_pos = scaler_info['var_pos']

    
    def run_the_network(self, start_pos: np.ndarray, start_vel: np.ndarray, t, verbose=False):
        
        if self.architecture == 'node' or self.architecture == 'recurrent':
            edge_idx_lst = []

            edge_idx_tsr = self.search_for_neighbor(start_pos,
                                                    self.nbr_searcher,
                                                    self.nbrlst_to_edge_mask,
                                                    'all')
            edge_idx_lst += [edge_idx_tsr]

            integration_time = torch.arange(t).to('cuda')
            integration_time = integration_time.float()

            start_vel = torch.from_numpy(start_vel).float().cuda()
            start_vel = self.scale(start_vel, self.train_data_scaler_vel).to('cuda')         
            start_pos = torch.from_numpy(start_pos).float().cuda()
            start_pos = self.scale(start_pos, self.train_data_scaler_pos).to('cuda')  

        with torch.no_grad():
            if self.architecture == 'gnode':
                pos_res, vel_res, emb_res = self.pnet_model([start_pos],
                                    [start_vel],
                                    integration_time,
                                    )
                pos_res = np.squeeze(denormalize_pos(pos_res).detach().cpu().numpy())
                vel_res = np.squeeze(denormalize_vel(vel_res).detach().cpu().numpy())

            if self.architecture == 'node' or self.architecture == 'gnode':
                pos_res, vel_res, emb_res = self.pnet_model([start_pos],
                                        edge_idx_lst,
                                        [start_vel],
                                        integration_time,
                                        )

                pos_res = np.squeeze(denormalize_pos(pos_res).detach().cpu().numpy())
                vel_res = np.squeeze(denormalize_vel(vel_res).detach().cpu().numpy())

            ## TO DO: fix normalization for recurrent

            if self.architecture == 'recurrent':
                pos_res = [start_pos.detach().cpu().numpy()]
                next_pos, start_vel, emb = self.pnet_model([start_pos],
                                        edge_idx_lst,
                                        [start_vel],
                                        integration_time,
                                        )
                for i in range (t-1):
                    emb = self.pnet_model.recurrent(emb)
                    emb_new, pos_new, vel_new = self.pnet_model.decoder(emb)
                    pos_res.append(pos_new.squeeze(0).detach().cpu().numpy())
                
                pos_res = np.stack(pos_res)
                vel_res = np.squeeze(denormalize_acc(vel_res).detach().cpu().numpy())


            if self.architecture == 'latentode' or self.architecture == 'graphlatentode':

                integration_time = torch.arange(int(t/2)).to('cuda')
                integration_time = integration_time.float()

                pos_lst = [torch.from_numpy(arr).to('cuda').to(torch.float32) for arr in start_pos]
                vel_lst = [torch.from_numpy(arr).to('cuda').to(torch.float32) for arr in start_vel]
                vel_lst = self.scale(vel_lst, self.train_data_scaler_vel).to('cuda')
                pos_lst = self.scale(pos_lst, self.train_data_scaler_pos).to('cuda')


                fluid_edge_lst = []
                for i in range (len(pos_lst)):
                    pos_lst_np = pos_lst[i].detach().cpu().numpy()
                    fluid_edge_tsr = self.search_for_neighbor(pos_lst_np,
                                                            self.nbr_searcher,
                                                            self.nbrlst_to_edge_mask,
                                                            'all')
                    fluid_edge_lst += [fluid_edge_tsr]
                if self. architecture == 'latentode':
                    pos_res, vel_res, emb_res = self.pnet_model(pos_lst,
                                                fluid_edge_lst,
                                                vel_lst,
                                                integration_time,
                                                )
                if self.architecture == 'graphlatentode':
                    integration_time = pos_lst.size()[0]
                    node_res, edge_res = self.pnet_model(pos_lst,
                                        fluid_edge_lst,
                                        vel_lst,
                                        integration_time,
                                        )

                    pos_res = np.squeeze(self.denormalize_pos(node_res.permute(1, 0, 2).detach().cpu().numpy()))
                    #vel_res = np.squeeze(denormalize_vel(vel_res).detach().cpu().numpy())

        return pos_res

    def make_a_graph(self, pos):
        edge_idx_tsr = self.search_for_neighbor(pos,
                                                    self.nbr_searcher,
                                                    self.nbrlst_to_edge_mask,
                                                    'all')

        center_idx = edge_idx_tsr[0, :]  # [edge_num, 1]
        neigh_idx = edge_idx_tsr[1, :]
        graph_now = dgl.graph((neigh_idx, center_idx))

        return graph_now

    def get_temperature(self, two_pos): ## assuming all particles have the same mass, and the timestep between each step is the same, but it can be adjucted for different timesteps

        coeff = 15500.21605 ## for 2fs difference in particle positions
        displacement = two_pos[0] - two_pos[1]
        squared = displacement.pow(2).sum(dim=1)
        total_squared = squared.sum()
        temperature = total_squared*coeff
        return temperature ## this isn't exactly the temperature, but temperature times a constant coeff

    def get_temp_pairs(self, pos):
        lst =[]
        for i in range(pos.size()[0]-1):
            lst.append(self.get_temperature((pos[i], pos[i+1])))
        return torch.stack(lst)

    def custom_loss(self, y_true, y_pred, sigma=1.0):
        mse_loss = torch.mean((y_true - y_pred) ** 2)
        
        # Assuming a fixed standard deviation sigma for the normal distribution
        normal_loss = torch.mean((y_true - y_pred) ** 2) / (2 * sigma**2)
        # No need to add the constant term of the log-likelihood since it doesn't affect optimization
        
        # Mask for values within the range 97 to 103
        in_range = (y_pred >= 97) & (y_pred <= 103)
        
        # Apply normal loss within the range and MSE loss outside the range
        loss = torch.where(in_range, normal_loss, mse_loss)
        return torch.mean(loss)

    def scale(self, inp, scaler):
        scaled_sequence = []
        for seq in inp:
            seq = seq.squeeze()
            if seq.ndim == 1:
                seq = seq.reshape((-1, 1))  # Reshape 1D array to 2D
            b_pnum, dims = seq.shape
            seq = seq.detach().cpu()  # Move tensor to CPU
            scaler.partial_fit(seq.numpy())  # Convert to NumPy array
            scaled_seq = scaler.transform(seq.numpy())
            scaled_seq_tensor = torch.from_numpy(scaled_seq).float().view(b_pnum, dims)
            scaled_sequence.append(scaled_seq_tensor)
        return torch.stack(scaled_sequence)

    def denormalize_vel(self, vel):
        return self.denormalize(vel, self.training_var_vel, self.training_mean_vel)
    
    def denormalize_pos(self, pos):
        return self.denormalize(pos, self.training_var_pos, self.training_mean_pos)

    def denormalize(self, normalized, var, mean):
        return normalized * \
                np.sqrt(var) +\
                mean

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

    def training_step(self, batch, batch_nb):

        torch.cuda.empty_cache()

        pos_lst = self.scale(batch[0]['pos'], self.train_data_scaler_pos).to('cuda')
        vel_lst = self.scale(batch[0]['vel'], self.train_data_scaler_vel).to('cuda')

        print(pos_lst.size())

        integration_time = torch.arange(pos_lst.size()[0]).to('cuda')
        integration_time = integration_time.float()

        if self.architecture == 'gnode':
            start_pos = pos_lst[0]
            start_vel = vel_lst[0]
            
            BOX_SIZE = torch.tensor([27.27, 27.27, 27.27]).to('cuda')
            start_pos = torch.fmod(start_pos, BOX_SIZE) #this I added

            pos_res, vel_res, emb_res = self.pnet_model([start_pos],
                                    [start_vel],
                                    integration_time,
                                    )

        if self.architecture == 'node':
            start_pos = pos_lst[0]
            start_vel = vel_lst[0]
            
            ### encode the first position
            BOX_SIZE = torch.tensor([27.27, 27.27, 27.27]).to('cuda')
            start_pos = torch.fmod(start_pos, BOX_SIZE) #this I added
            start_pos_np = start_pos.detach().cpu().numpy()

            edge_idx_lst = []
            edge_idx_tsr = self.search_for_neighbor(start_pos_np,
                                                    self.nbr_searcher,
                                                    self.nbrlst_to_edge_mask,
                                                    'all')
            edge_idx_lst += [edge_idx_tsr]

            pos_res, vel_res, emb_res = self.pnet_model([start_pos],
                                    edge_idx_lst,
                                    [start_vel],
                                    integration_time,
                                    )

        if self.architecture == 'recurrent' or self.architecture == 'latentode' or self.architecture == 'graphlatentode':

            ### calculate the "true" emebeddings
            fluid_edge_lst = []
            for i in range (len(pos_lst)):
                pos_lst_np = pos_lst[i].detach().cpu().numpy()
                fluid_edge_tsr = self.search_for_neighbor(pos_lst_np,
                                                        self.nbr_searcher,
                                                        self.nbrlst_to_edge_mask,
                                                        'all')
                fluid_edge_lst += [torch.tensor(fluid_edge_tsr)]
            
        if self.architecture == 'graphlatentode':
            
            integration_time = pos_lst.size()[0]
            node_res, edge_res = self.pnet_model(pos_lst[0:int(integration_time/2)],
                                    fluid_edge_lst[0:int(integration_time/2)],
                                    vel_lst[0:int(integration_time/2)],
                                    int(integration_time/2),
                                    )
            c = 1
            edge_gt = []
            for i in range(int(integration_time/2),integration_time):
                edge_gt.append(self.pnet_model.get_edge_weight_matrix(pos_lst[i], fluid_edge_lst[i], c))

            edge_gt = torch.stack(edge_gt)

            node_loss = nn.MSELoss()(pos_lst[int(integration_time/2):], node_res.permute(1, 0, 2))
            edge_loss = nn.MSELoss()(edge_res, edge_gt.view_as(edge_res).cuda())
            loss = node_loss# + edge_loss
            self.log('cord_loss', node_loss, on_step=True, prog_bar=True, logger=True)
            self.log('edge_loss', edge_loss, on_step=True, prog_bar=True, logger=True)
            return loss

        if self.architecture == 'recurrent' or self.architecture == 'latentode':  
            pos_res, vel_res, force_res = self.pnet_model(pos_lst,
                                    fluid_edge_lst,
                                    vel_lst,
                                    integration_time,
                                    )

        #with torch.no_grad():
        #    emb_lst = self.pnet_model.encoder(pos_lst, fluid_edge_lst, vel_lst).view_as(emb_res)

        if self.loss_fn == 'mae':
            cord_loss = nn.L1Loss()(pos_lst, pos_res)
            #rec_loss = nn.L1Loss()(graphem1, graphem2)
            #force_loss = nn.L1Loss()(pred_forces, forces_gt)
            vel_loss = nn.L1Loss()(vel_lst, vel_res)
            #emb_loss = nn.L1Loss()(emb_lst, emb_res)
        else:
            cord_loss = nn.MSELoss()(pos_lst, pos_res)
            #rec_loss = nn.MSELoss()(graphem1, graphem2)
            #force_loss = nn.MSELoss()(pred_forces, forces_gt)
            vel_loss = nn.MSELoss()(vel_lst, vel_res)
            #emb_loss = nn.L1Loss()(emb_lst, emb_res)

        #regularization_loss = (torch.mean(graphem1)).abs()

        self.training_mean_pos = self.train_data_scaler_pos.mean_
        self.training_var_pos = self.train_data_scaler_pos.var_

        self.training_mean_vel = self.train_data_scaler_vel.mean_
        self.training_var_vel = self.train_data_scaler_vel.var_
        temp = self.get_temp_pairs(pos_res)
        temp_true = torch.full_like(temp, fill_value=float(100))

        temp_loss = nn.MSELoss()(temp, temp_true)
        temp_loss = self.custom_loss(temp, torch.full_like(temp, fill_value=float(100)))

        #loss = cord_loss + rec_loss + 0.5/regularization_loss

        #conservative_loss = (torch.mean(pred)).abs() #conservative loss penalizes size of the predictions (remove it, leave it?)
        #loss = cord_loss #+ 0.1*vel_loss #+ LAMBDA2 * conservative_loss
        loss = cord_loss + vel_loss #+ temp_loss +0.1*emb_loss

        self.log('cord_loss', cord_loss, on_step=True, prog_bar=True, logger=True)
        self.log('vel_loss', vel_loss, on_step=True, prog_bar=True, logger=True)
        #self.log('emb_loss', emb_loss, on_step=True, prog_bar=True, logger=True)
        self.log('temp_loss', temp_loss, on_step=True, prog_bar=True, logger=True)
        self.log('temp', torch.mean(temp), on_step=True, prog_bar=True, logger=True)
        #self.log('force_loss', force_loss, on_step=True, prog_bar=True, logger=True)
        #self.log('graph rec loss', rec_loss, on_step=True, prog_bar=True, logger=True)
        self.log(f'actual loss:{self.loss_fn}', loss, on_step=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
            optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            sched = StepLR(optim, step_size=10, gamma=0.001**(10/self.epoch_num)) ## changed 5 to 2
            return [optim], [sched]

    def train_dataloader(self):

        dataset = sequence_of_pos(dataset_path=os.path.join(self.data_dir, 'lj_data_20ts'),
                               seed_num=10,
                               mode='train')

        return DataLoader(dataset, batch_size=1, shuffle = True, pin_memory=False)
        

    def val_dataloader(self):

        dataset = sequence_of_pos(dataset_path=os.path.join(self.data_dir, 'lj_data_20ts'),
                               seed_num=10,
                               mode='test')

        return DataLoader(dataset, batch_size=1, shuffle = True, pin_memory=False)


    def validation_step(self, batch, batch_nb):

        torch.cuda.empty_cache()

        #if network_has_been_trained == True:
        with torch.no_grad():

            pos_lst = self.scale(batch[0]['pos'], self.train_data_scaler_pos).to('cuda')
            vel_lst = self.scale(batch[0]['vel'], self.train_data_scaler_vel).to('cuda')

            #gt_pos = torch.cat(pos_lst, dim=0)
            #gt_vel = torch.cat(vel_lst, dim=0)

            integration_time = torch.arange(pos_lst.size()[0]).to('cuda')
            integration_time = integration_time.float()

            if self.architecture == 'gnode':
                start_pos = pos_lst[0]
                start_vel = vel_lst[0]
                
                BOX_SIZE = torch.tensor([27.27, 27.27, 27.27]).to('cuda')
                start_pos = torch.fmod(start_pos, BOX_SIZE) #this I added

                pos_res, vel_res, emb_res = self.pnet_model([start_pos],
                                        [start_vel],
                                        integration_time,
                                        )

            if self.architecture == 'node':

                start_pos = pos_lst[0]
                start_vel = vel_lst[0]
                
                ### encode the first position
                BOX_SIZE = torch.tensor([27.27, 27.27, 27.27]).to('cuda')
                start_pos = torch.fmod(start_pos, BOX_SIZE) #this I added
                start_pos_np = start_pos.detach().cpu().numpy()

                edge_idx_lst = []
                edge_idx_tsr = self.search_for_neighbor(start_pos_np,
                                                        self.nbr_searcher,
                                                        self.nbrlst_to_edge_mask,
                                                        'all')
                edge_idx_lst += [edge_idx_tsr]

                pos_res, vel_res, emb_res = self.pnet_model([start_pos],
                                        edge_idx_lst,
                                        [start_vel],
                                        integration_time,
                                        )
            
            if self.architecture == 'recurrent' or self.architecture == 'latentode' or self.architecture == 'graphlatentode':
                
                ### calculate the "true" emebeddings
                fluid_edge_lst = []
                for i in range (len(pos_lst)):
                    pos_lst_np = pos_lst[i].detach().cpu().numpy()
                    fluid_edge_tsr = self.search_for_neighbor(pos_lst_np,
                                                            self.nbr_searcher,
                                                            self.nbrlst_to_edge_mask,
                                                            'all')
                    fluid_edge_lst += [fluid_edge_tsr]
            
            if self.architecture == 'graphlatentode':

                integration_time = pos_lst.size()[0]
                node_res, edge_res = self.pnet_model(pos_lst[0:int(integration_time/2)],
                                        fluid_edge_lst[0:int(integration_time/2)],
                                        vel_lst[0:int(integration_time/2)],
                                        int(integration_time/2),
                                        )

                edge_gt = []
                c=1
                for i in range(int(integration_time/2),integration_time):
                    edge_gt.append(self.pnet_model.get_edge_weight_matrix(pos_lst[i], fluid_edge_lst[i], c))

                edge_gt = torch.stack(edge_gt)

                node_loss = nn.MSELoss()(pos_lst[int(integration_time/2):], node_res.permute(1, 0, 2))

                edge_loss = nn.MSELoss()(edge_res, edge_gt.view_as(edge_res).cuda())

                self.log('val node', node_loss, on_step=True, prog_bar=True, logger=True)
                self.log('val edge', edge_loss, on_step=True, prog_bar=True, logger=True)
                self.log('val all', edge_loss+node_loss, on_step=True, prog_bar=True, logger=True)
                return

            if self.architecture == 'recurrent' or self.architecture == 'latentode':
            
                pos_res, vel_res, force_res = self.pnet_model(pos_lst,
                                        fluid_edge_lst,
                                        vel_lst,
                                        integration_time,
                                        )

            #with torch.no_grad():
            #    emb_lst = self.pnet_model.encoder(pos_lst, fluid_edge_lst, vel_lst).view_as(emb_res)

            if self.loss_fn == 'mae':
                cord_loss = nn.L1Loss()(pos_lst, pos_res)
                #rec_loss = nn.L1Loss()(graphem1, graphem2)
                #force_loss = nn.L1Loss()(pred_forces, forces_gt)
                vel_loss = nn.L1Loss()(vel_lst, vel_res)
                #emb_loss = nn.L1Loss()(emb_lst, emb_res)
            else:
                cord_loss = nn.MSELoss()(pos_lst, pos_res)
                #rec_loss = nn.MSELoss()(graphem1, graphem2)
                #force_loss = nn.MSELoss()(pred_forces, forces_gt)
                vel_loss = nn.MSELoss()(vel_lst, vel_res)
                #emb_loss = nn.L1Loss()(emb_lst, emb_res)

            #regularization_loss = (torch.mean(graphem1)).abs()

            #temp = self.get_temp_pairs(pos_res)
            #temp_true = torch.full_like(temp, fill_value=float(100))

            #temp_loss = nn.MSELoss()(temp, temp_true)
            #temp_loss = self.custom_loss(temp, torch.full_like(temp, fill_value=float(100)))

            #conservative_loss = (torch.mean(pred)).abs() #conservative loss penalizes size of the predictions (remove it, leave it?)
            #loss = cord_loss #+ 0.1*vel_loss #+ LAMBDA2 * conservative_loss

            self.log('val_cord_loss', cord_loss, on_step=True, prog_bar=True, logger=True)
            self.log('val_vel_loss', vel_loss, on_step=True, prog_bar=True, logger=True)
            #self.log('force_loss', force_loss, on_step=True, prog_bar=True, logger=True)
            #self.log('graph rec loss', rec_loss, on_step=True, prog_bar=True, logger=True)
            self.log(f'val loss:{self.loss_fn}', loss, on_step=True, prog_bar=True, logger=True)

class ModelCheckpointAtEpochEnd(pl.Callback):
    """
       Save a checkpoint at epoch end
    """
    def __init__(
            self,
            filepath,
            save_step_frequency,
            prefix="checkpoint",
            prefix1="encoder_checkpoint",
            prefix2="decoder_checkpoint",
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
        """
        self.filepath = filepath
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: MDSimNet):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if epoch % self.save_step_frequency == 0 or epoch == pl_module.epoch_num -1:
            filename = os.path.join(self.filepath, f"{self.prefix}_{epoch}.ckpt")
            scaler_filename = os.path.join(self.filepath, f"scaler_{epoch}.npz")

            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            np.savez(scaler_filename,
                     mean_vel=pl_module.training_mean_vel,
                     var_vel=pl_module.training_var_vel,
                     mean_pos=pl_module.training_mean_pos,
                     var_pos=pl_module.training_var_pos,
                     )
            # joblib.dump(pl_module.train_data_scaler, scaler_filename)


def train_model(args):
    lr = args.lr
    num_gpu = args.num_gpu
    check_point_dir = args.cp_dir
    min_epoch = args.min_epoch
    max_epoch = args.max_epoch
    weight_ckpt = args.state_ckpt_dir
    batch_size = args.batch_size
    mode = args.mode
    architecture = args.architecture
    wandb_logger = WandbLogger()

    model = MDSimNet(epoch_num=max_epoch,
                                 num_device=num_gpu if num_gpu != -1 else 1,
                                 learning_rate=lr,
                                 model_weights_ckpt=weight_ckpt,
                                 batch_size=batch_size,
                                 args=args)
    cwd = os.getcwd()
    model_check_point_dir = os.path.join(cwd, check_point_dir)
    os.makedirs(model_check_point_dir, exist_ok=True)
    epoch_end_callback = ModelCheckpointAtEpochEnd(filepath=model_check_point_dir, save_step_frequency=5)
    checkpoint_callback = pl.callbacks.ModelCheckpoint()
    #early_stop_callback = EarlyStopping(monitor='val coordinates (mse)', patience=5, verbose=False, mode="min")

    trainer = Trainer(gpus=num_gpu,
                      callbacks=[epoch_end_callback, checkpoint_callback],
                      min_epochs=min_epoch,
                      max_epochs=max_epoch,
                      amp_backend='apex',
                      amp_level='O1',
                      benchmark=True,
                      distributed_backend='ddp',
                      logger=wandb_logger,
                      resume_from_checkpoint='/home/guests/lana_frkin/GAMDplus/code/LJ/model_ckpt/EVERYTHING_20s/checkpoint_20.ckpt',
                      )
    trainer.fit(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_epoch', default = 5000, type=int)
    parser.add_argument('--max_epoch', default = 5000, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--cp_dir', default='./model_ckpt/EVERYTHING_20s_just_node')
    parser.add_argument('--state_ckpt_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--encoding_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--edge_embedding_dim', default=32, type=int)
    parser.add_argument('--drop_edge', action='store_true', default = True)
    parser.add_argument('--use_layer_norm', action='store_true', default = False)
    parser.add_argument('--disable_rotate_aug', dest='rotate_aug', default=True, action='store_false')
    parser.add_argument('--data_dir', default='./md_dataset')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--num_gpu', default=-1, type=int)
    parser.add_argument('--architecture', default='latentode', type=str)
    parser.add_argument('--mode', default='train', type=str)
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()