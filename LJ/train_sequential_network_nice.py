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

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from nn_module import SimpleMDNetNew
from train_utils_seq import Sequential_data
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

import torch
from torchdyn.models import NeuralODE

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        # Define your neural network layers here
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        # Implement the dynamics of your system here
        # You can use the defined layers and return dx/dt
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out

# In your NeuralODETimeSeries class
class NeuralODETimeSeries(pl.LightningModule):
    def __init__(self, num_layers, hidden_size):
        super(NeuralODETimeSeries, self).__init__()
        self.func = ODEFunc()  # Instantiate your ODE function
        self.ode = NeuralODE(self.func, solver='dopri5')
        self.fc = torch.nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x):
        z = self.ode(x)
        return self.fc(z[1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        x = x.T
        x = x.unsqueeze(0)
        x = x.T
        y = y.T
        y = y.unsqueeze(0)
        y = y.T

        y_pred = self(x.T)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        dataset = Sequential_data(sample_num=1)[0]
        
        return DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)


def main():

    # Create a PyTorch Lightning model
    model = NeuralODETimeSeries(num_layers=3, hidden_size=128)

    # Create a PyTorch Lightning Trainer and train the model
    trainer = pl.Trainer(max_epochs=100, gpus=1)
    trainer.fit(model)

if __name__ == "__main__":
    main()




"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch
import torch.nn as nn
import pytorch_lightning as pl

class MyRNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):

        out, hidden_state = self.rnn(x, h0)

        y_pred = self.fc(out)
        
        return y_pred, hidden_state

    def training_step(self, batch, batch_idx):
        x_batch = batch[0]

        print(type(x_batch))
        print(x_batch[0].size())

        # Get the model's predictions for the entire sequence
        y_pred, hidden_state = self(x_batch, hidden_state)
        print("x_batch shape:", x_batch.shape)
        print("y_pred shape in training_step:", y_pred.shape)

        # Compute the mean squared error loss between predictions and actual values
        loss = nn.MSELoss()(y_pred, x_batch)

        # Log the training loss
        self.log("train_loss", loss)

        return loss

    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    def dataloader(self):
        # Replace with your dataset and DataLoader setup
        def dataloader(self):
        # Replace with your dataset and DataLoader setup
        dataset = Sequential_data(sample_num=1) 
        return DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)

# Create the model
model = MyRNN(input_size=128, hidden_size=64, num_layers=1, output_size=128)


# Create a PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=100, gpus=1)  # Adjust max_epochs and other options as needed

# Train the model using the dataloader method
trainer.fit(model, model.dataloader())

"""