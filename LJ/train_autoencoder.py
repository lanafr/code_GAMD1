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
from nn_module import GNNAutoencoder
from train_utils import Graphs_data
from graph_utils import NeighborSearcher, graph_network_nbr_fn
import time
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
LAMBDA2 = 1e-3

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
    model = GNNAutoencoder(**param_dict)

    if ckpt is not None:
        print('Loading model weights from: ', ckpt)
        model.load_state_dict((torch.load(ckpt)))
    return model


class AutoencoderNetLightning(pl.LightningModule):
    def __init__(self, args, num_device=1, epoch_num=100, batch_size=1, learning_rate=3e-4, log_freq=1000,
                 model_weights_ckpt=None, scaler_ckpt=None):
        super(AutoencoderNetLightning, self).__init__()
        self.pnet_model = build_model(args, model_weights_ckpt)
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_device = num_device
        self.log_freq = log_freq
        self.train_data_scaler = StandardScaler()
        self.training_mean = np.array([0.])
        self.training_var = np.array([1.])

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
            self.training_mean = scaler_info['mean']
            self.training_var = scaler_info['var']

    
    def do_the_autoencoding(self, g:dgl.DGLGraph) -> dgl.DGLGraph:
        nbr_start = time.time()
        nbr_end = time.time()

        graph_old = g.to('cuda')
        force_start = time.time()
        graph_new = self.pnet_model(graph_old
                               )
        force_end = time.time()
        if verbose:
            print('=============================================')
            print(f'Nbr search used time: {nbr_end - nbr_start}')
            print(f'Next pos prediction used time: {force_end - force_start}')

        graph_new = graph_new.detach().cpu().numpy()

        return graph_new

    def training_step(self, batch):
        
        old_graph = batch[0][0]
        print(type(batch))
        print(type(old_graph))

        new_graph = self.do_the_autoencoding(old_graph
                               )

        pred = new_graph.ndata['e']
        gt = old_graph.ndata['e']

        if self.loss_fn == 'mae':
            loss = nn.L1Loss()(pred, gt)
        else:
            loss = nn.MSELoss()(pred, gt)

        first_loss = loss

        conservative_loss = (torch.mean(pred)).abs()
        loss = loss + LAMBDA2 * conservative_loss

        self.log('total loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log(f'{self.loss_fn} loss', first_loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sched = StepLR(optim, step_size=5, gamma=0.001**(5/self.epoch_num))
        return [optim], [sched]

    def collate_dgl_graphs(batch):

        batched_graph = dgl.batch(batch)

        return batched_graph

    def train_dataloader(self):
        dataset = Graphs_data(dataset_path=os.path.join(self.data_dir),
                               sample_num=9055, #this I changed
                               case_prefix='graphs_to_train',
                               mode='train')

        #return DataLoader(dataset,num_workers=2, batch_size=1, shuffle=False, collate_fn=self.collate_dgl_graphs)
        distributed_sampler = DistributedSampler(dataset, seed=0)
        return dgl.dataloading.GraphDataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=2, sampler=distributed_sampler)

    def val_dataloader(self):
        dataset = Graphs_data(dataset_path=os.path.join(self.data_dir),
                               sample_num=9055, #this I changed
                               case_prefix='graphs_to_train',
                               mode='test')

        #return DataLoader(dataset,num_workers=2, batch_size=1, shuffle=False, collate_fn=self.collate_dgl_graphs)
        distributed_sampler = DistributedSampler(dataset, seed=0)
        return dgl.dataloading.GraphDataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=2, sampler=distributed_sampler)


    def validation_step(self, batch, batch_nb):
        with torch.no_grad():

            old_graph = batch[0][0]

            new_graph = self.do_the_autoencoding(old_graph
                               )

            pred = new_graph.ndata['e']
            gt = old_graph.ndata['e']

            mse = nn.MSELoss()(pred, gt)
            mae = nn.L1Loss()(pred, gt)

            self.log('val outlier', outlier_ratio, prog_bar=True, logger=True)
            self.log('val mse', mse, prog_bar=True, logger=True)
            self.log('val mae', mae, prog_bar=True, logger=True)


class ModelCheckpointAtEpochEnd(pl.Callback):
    """
       Save a checkpoint at epoch end
    """
    def __init__(
            self,
            filepath,
            save_step_frequency,
            prefix="checkpoint",
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
        """
        self.filepath = filepath
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: AutoencoderNetLightning):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if epoch % self.save_step_frequency == 0 or epoch == pl_module.epoch_num -1:
            filename = os.path.join(self.filepath, f"{self.prefix}_{epoch}.ckpt")
            scaler_filename = os.path.join(self.filepath, f"scaler_{epoch}.npz")

            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            np.savez(scaler_filename,
                     mean=pl_module.training_mean,
                     var=pl_module.training_var,
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
    wandb_logger = WandbLogger()

    model = AutoencoderNetLightning(epoch_num=max_epoch,
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
    parser.add_argument('--min_epoch', default=30, type=int)
    parser.add_argument('--max_epoch', default=30, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--cp_dir', default='./model_ckpt_autoencoder/autoencoder_for_graphs1')
    parser.add_argument('--state_ckpt_dir', default=None, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--encoding_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--edge_embedding_dim', default=1, type=int)
    parser.add_argument('--drop_edge', action='store_true')
    parser.add_argument('--use_layer_norm', action='store_true')
    parser.add_argument('--disable_rotate_aug', dest='rotate_aug', default=True, action='store_false')
    parser.add_argument('--data_dir', default='./graphs_to_train')
    parser.add_argument('--loss', default='mae')
    parser.add_argument('--num_gpu', default=-1, type=int)
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()