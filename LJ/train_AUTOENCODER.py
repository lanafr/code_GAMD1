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
from SIMPLER_GNN import *
from train_utils_seq import *
from scipy.spatial import KDTree

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

def center_positions(pos):
    offset = np.mean(pos, axis=0)
    return pos - offset, offset


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
                  }

    print("Using following set of hyper-parameters")
    print(param_dict)
    model = Autoencoder(**param_dict)

    if ckpt is not None:
        print('Loading model weights from: ', ckpt)
        model.load_state_dict((torch.load(ckpt)))
    return model


class ParticleAutoencoder(pl.LightningModule):
    def __init__(self, args, num_device=1, epoch_num=100, batch_size=1, learning_rate=3e-4, log_freq=1000,
                 model_weights_ckpt=None, scaler_ckpt=None):
        super(ParticleAutoencoder, self).__init__()
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
            self.training_mean_pos = scaler_info['mean_pos']
            self.training_var_pos = scaler_info['var_pos']
            self.training_mean_vel = scaler_info['mean_vel']
            self.training_var_vel = scaler_info['var_vel']

    def scale_force(self, force, scaler):
        b_pnum, dims = force.shape
        force_flat = force.reshape((-1, 1))
        scaler.partial_fit(force_flat)
        force = torch.from_numpy(scaler.transform(force_flat)).float().view(b_pnum, dims)
        return force

    def full_autoencoder_vel(self, ins):
        return self.denormalize(ins, self.training_var_vel, self.training_mean_vel)

    def full_autoencoder_pos(self, ins):
        return self.denormalize(ins, self.training_var_pos, self.training_mean_pos)

    def denormalize(self, normalized_force, var, mean):
        return normalized_force * \
                np.sqrt(var) +\
                mean

    def embed_pos(self, pos: np.ndarray, vel: np.ndarray, verbose=False):
        nbr_start = time.time()
        edge_idx_tsr = self.search_for_neighbor(pos,
                                                self.nbr_searcher,
                                                self.nbrlst_to_edge_mask,
                                                'all')
        nbr_end = time.time()
        # enforce periodic boundary
        pos = np.mod(pos, np.array(BOX_SIZE))
        pos = torch.from_numpy(pos).float().cuda()
        vel = torch.from_numpy(vel).float().cuda()
        force_start = time.time()
        #pred, emb, forces = self.pnet_model([pos],
        pred, emb, vel = self.pnet_model([pos],
                               [edge_idx_tsr],
                               [vel]
                               )
        force_end = time.time()
        if verbose:
            print('=============================================')
            print(f'Nbr search used time: {nbr_end - nbr_start}')
            print(f'Next pos prediction used time: {force_end - force_start}')

        pred = pred.detach().cpu().numpy()

        return pred, emb, vel

    def make_a_graph(self,pos):
        edge_idx_tsr = self.search_for_neighbor(pos,
                                                    self.nbr_searcher,
                                                    self.nbrlst_to_edge_mask,
                                                    'all')

        center_idx = edge_idx_tsr[0, :]  # [edge_num, 1]
        neigh_idx = edge_idx_tsr[1, :]
        graph_now = dgl.graph((neigh_idx, center_idx))

        return graph_now

    def autoencode(self, start_pos, start_vel):

        gt = start_pos
        vel = start_vel

        edge_idx_lst = []

        gt = np.mod(gt, BOX_SIZE) #this I added
        gt_np = gt
        gt = self.scale_force(gt, self.train_data_scaler_pos).cuda()
        vel = self.scale_force(vel, self.train_data_scaler_vel).cuda()

        print(self.train_data_scaler_pos)

        print(self.training_var_pos)

        print(self.training_mean_pos)


        edge_idx_tsr = self.search_for_neighbor(gt_np,
                                                self.nbr_searcher,
                                                self.nbrlst_to_edge_mask,
                                                'all')
        edge_idx_lst += [edge_idx_tsr]

        with torch.no_grad():

            res_pos, res_vel, res_forces = self.pnet_model([gt],
                                    edge_idx_lst,
                                    [vel]
                                    )

        print("Loss inside:")
        print(nn.MSELoss()(res_pos.squeeze(), gt))

        res_vel = self.full_autoencoder_vel(res_vel.squeeze().detach().cpu().numpy())
        res_pos = self.full_autoencoder_pos(res_pos.squeeze().detach().cpu().numpy())

        return res_pos


    def decode_the_sequence(self, sequence_embeddings: torch.Tensor, t):
        trajectory = []
        
        for i in range(t):
            #graph_emb, pos_next, forces = self.pnet_model.gdecoder_MLP(sequence_embeddings[i])
            graph_emb, pos_next, vel = self.pnet_model.gdecoder_MLP(sequence_embeddings[i])
            
            pos_next = pos_next.detach().cpu().numpy()
            trajectory.append(pos_next)

        return trajectory


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
        gt_lst = batch ['pos']
        edge_idx_lst = []
        vel_lst = batch['vel']
        
        for b in range(len(gt_lst)):
            gt = gt_lst[b]
            vel = vel_lst[b]

            gt = np.mod(gt, BOX_SIZE) #this I added
            gt_np = gt
            gt = self.scale_force(gt, self.train_data_scaler_pos).cuda()
            vel = self.scale_force(vel, self.train_data_scaler_vel).cuda()


            gt_lst[b] = gt
            vel_lst[b] = vel #this I added

            edge_idx_tsr = self.search_for_neighbor(gt_np,
                                                    self.nbr_searcher,
                                                    self.nbrlst_to_edge_mask,
                                                    'all')
            edge_idx_lst += [edge_idx_tsr]

        gt = torch.cat(gt_lst, dim=0)
        vel = torch.cat(vel_lst, dim=0)

        res_pos, res_vel, res_forces = self.pnet_model(gt_lst,
                                   edge_idx_lst,
                                   vel_lst
                                   )

        self.training_mean_pos = self.train_data_scaler_pos.mean_
        self.training_var_pos = self.train_data_scaler_pos.var_

        self.training_mean_vel = self.train_data_scaler_vel.mean_
        self.training_var_vel = self.train_data_scaler_vel.var_

        res_pos = res_pos.permute(2, 0, 1).contiguous().view(res_pos.size(2), -1).permute(1, 0)
        res_vel = res_vel.permute(2, 0, 1).contiguous().view(res_vel.size(2), -1).permute(1, 0)

        if self.loss_fn == 'mae':
            cord_loss = nn.L1Loss()(res_pos, gt)
            #rec_loss = nn.L1Loss()(graphem1, graphem2)
            #force_loss = nn.L1Loss()(res_forces, forces)
            vel_loss = nn.L1Loss()(res_vel, vel)
        else:
            cord_loss = nn.MSELoss()(res_pos, gt)
            #rec_loss = nn.MSELoss()(graphem1, graphem2)
            #force_loss = nn.MSELoss()(res_forces, forces)
            vel_loss = nn.MSELoss()(res_vel, vel)

        regularization_loss = (torch.mean(res_vel)).abs()

        #loss = cord_loss + rec_loss + 0.5/regularization_loss

        variance_loss = torch.mean(torch.var(res_pos, dim=1))

        conservative_loss = (torch.mean(res_pos)).abs() #conservative loss penalizes size of the predictions (remove it, leave it?)
        loss = cord_loss + (1/variance_loss)# + vel_loss + LAMBDA2*regularization_loss # + vel_loss #+ LAMBDA2 conservative_loss

        self.log('cord_loss', cord_loss, on_step=True, prog_bar=True, logger=True)
        self.log('variance', torch.sqrt(variance_loss), on_step=True, prog_bar=True, logger=True)
        self.log('vel_loss', vel_loss, on_step=True, prog_bar=True, logger=True)
        #self.log('forces_loss', force_loss, on_step=True, prog_bar=True, logger=True)
        self.log(f'actual loss:{self.loss_fn}', loss, on_step=True, prog_bar=True, logger=True)


        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sched = StepLR(optim, step_size=10, gamma=0.001**(5/self.epoch_num)) ## changed 5 to 2
        return [optim], [sched]

    def train_dataloader(self):
        dataset = LJDataNew(dataset_path=os.path.join(self.data_dir, 'lj_data'),
                               sample_num=999, #this I changed
                               case_prefix='data_',
                               seed_num=10,
                               mode='train')

        return DataLoader(dataset, num_workers=2, batch_size=self.batch_size, shuffle=True, # I changed shuffle from True to False to see how it effects the training and validation
                          collate_fn=
                          lambda batches: {
                              'pos': [batch['pos'] for batch in batches],
                              'forces': [batch['forces'] for batch in batches],
                              'vel': [batch['vel'] for batch in batches]
                          })
        
    def val_dataloader(self):
        dataset = LJDataNew(dataset_path=os.path.join(self.data_dir, 'lj_data'),
                               sample_num=999, #this I changed
                               case_prefix='data_',
                               seed_num=10,
                               mode='test')

        return DataLoader(dataset, num_workers=2, batch_size=self.batch_size, shuffle=True,
                          collate_fn=
                          lambda batches: {
                              'pos': [batch['pos'] for batch in batches],
                              'forces': [batch['forces'] for batch in batches],
                              'vel': [batch['vel'] for batch in batches],
                          })


    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            gt_lst = batch ['pos']
            edge_idx_lst = []
            vel_lst = batch['vel']
            
            for b in range(len(gt_lst)):
                gt = gt_lst[b]
                vel = vel_lst[b]

                gt = np.mod(gt, BOX_SIZE) #this I added
                gt_np = gt
                gt = self.scale_force(gt, self.train_data_scaler_pos).cuda()
                vel = self.scale_force(vel, self.train_data_scaler_vel).cuda()


                gt_lst[b] = gt
                vel_lst[b] = vel #this I added

                edge_idx_tsr = self.search_for_neighbor(gt_np,
                                                        self.nbr_searcher,
                                                        self.nbrlst_to_edge_mask,
                                                        'all')
                edge_idx_lst += [edge_idx_tsr]

            gt = torch.cat(gt_lst, dim=0)
            vel = torch.cat(vel_lst, dim=0)

            res_pos, res_vel, res_forces = self.pnet_model(gt_lst,
                                    edge_idx_lst,
                                    vel_lst
                                    )

            res_pos = res_pos.permute(2, 0, 1).contiguous().view(res_pos.size(2), -1).permute(1, 0)
            res_vel = res_vel.permute(2, 0, 1).contiguous().view(res_vel.size(2), -1).permute(1, 0)

            mse = nn.MSELoss()(res_pos, gt)
            mse_vel = nn.MSELoss()(res_vel, vel)
            #mse_force = nn.MSELoss()(res_forces, forces)
            mae = nn.L1Loss()(res_pos, gt)
            mae_vel = nn.L1Loss()(res_vel, vel)
            #mae_force = nn.L1Loss()(res_forces, forces)

            self.log('val coordinates (mse)', mse, prog_bar=True, logger=True)
            self.log('val coordinates (mae)', mae, prog_bar=True, logger=True)
            self.log('val velocity (mae)', mae_vel, prog_bar=True, logger=True)
            self.log('val velocity (mse)', mse_vel, prog_bar=True, logger=True)
            #self.log('val forces (mae)', mae_force, prog_bar=True, logger=True)
            #self.log('val forces (mse)', mse_force, prog_bar=True, logger=True)
            
            #self.log('val graphs (mae)', mae_for_graphs, prog_bar=True, logger=True)
            #self.log('val graphs (mse)', mse_for_graphs, prog_bar=True, logger=True)


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
        self.prefix1 = prefix1
        self.prefix2 = prefix2

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: ParticleAutoencoder):
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

            ### different checkpoints saved for Encoder and Decoder
            filename1 = os.path.join(self.filepath, f"{self.prefix1}_{epoch}.ckpt")
            ckpt_path1 = os.path.join(trainer.checkpoint_callback.dirpath, filename1)
            torch.save(pl_module.pnet_model.encoder.state_dict(), ckpt_path1)

            filename2 = os.path.join(self.filepath, f"{self.prefix2}_{epoch}.ckpt")
            ckpt_path2 = os.path.join(trainer.checkpoint_callback.dirpath, filename2)
            torch.save(pl_module.pnet_model.decoder.state_dict(), ckpt_path2)


def train_model(args):
    lr = args.lr
    num_gpu = args.num_gpu
    check_point_dir = args.cp_dir
    min_epoch = args.min_epoch
    max_epoch = args.max_epoch
    weight_ckpt = args.state_ckpt_dir
    batch_size = args.batch_size
    wandb_logger = WandbLogger()

    model = ParticleAutoencoder(epoch_num=max_epoch,
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
                      )
    trainer.fit(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_epoch', default=40, type=int)
    parser.add_argument('--max_epoch', default=40, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--cp_dir', default='./model_ckpt/AUTOENCODER_50ts(cords)_NNConv_512')
    parser.add_argument('--state_ckpt_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--encoding_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--edge_embedding_dim', default=32, type=int)
    parser.add_argument('--drop_edge', action='store_true', default = True)
    parser.add_argument('--use_layer_norm', action='store_true', default = False)
    parser.add_argument('--disable_rotate_aug', dest='rotate_aug', default=True, action='store_false')
    parser.add_argument('--data_dir', default='./md_dataset')
    parser.add_argument('--loss', default='mae')
    parser.add_argument('--num_gpu', default=-1, type=int)
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()