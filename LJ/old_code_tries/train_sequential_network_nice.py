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
from train_utils_seq import Sequential_data, Some_seq_data
from graph_utils import NeighborSearcher, graph_network_nbr_fn
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # just to test if it works w/o gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

## try number 100

import torch
import torchcde

from heavyballNODE import *

import wandb

seqlen = 64
data = Some_seq_data(1)

wandb.init(project="NeuralODE jej", name="Sequential model")


class tempf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, in_channels)
        self.dense2 = nn.Linear(in_channels, out_channels)
        # self.dense3 = nn.Linear(out_channels, out_channels)

    def forward(self, h, x):
        out = self.dense1(x)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        # out = self.dense3(out)
        return out


class temprnn(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, res=False, cont=False):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels + 2 * nhidden, 2 * nhidden)
        self.dense2 = nn.Linear(2 * nhidden, 2 * nhidden)
        self.dense3 = nn.Linear(2 * nhidden, 2 * out_channels)
        self.cont = cont
        self.res = res

    def forward(self, h, x):
        print(h[:, 0].size())
        print(x.size())
        out = torch.cat([h[:, 0], h[:, 1], x], dim=1)
        print(h[:, 0].size())
        print(x.size())
        out = self.dense1(out)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out).reshape(h.shape)
        out = out + h
        return out


class MODEL(nn.Module):
    def __init__(self, res=False, cont=False):
        super(MODEL, self).__init__()
        nhid = 256
        self.cell = HeavyBallNODE(tempf(nhid, nhid), corr=0, corrf=True)
        # self.cell = HeavyBallNODE(tempf(nhid, nhid))
        #self.try1 = NODEintegrate(self.cell)
        self.rnn = temprnn(8, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, (2, nhid), None, tol=1e-7)
        self.outlayer = nn.Linear(nhid, 8)

    def forward(self, t, x):
        x = torch.tensor(x).to('cuda')
        t = torch.tensor(t).to('cuda')
        out = self.ode_rnn(t, x, retain_grad=True)[0]
        #out = self.try1(t, x)
        out = self.outlayer(out[:, :, 0])[1:]
        return out


def main():
    data = Some_seq_data(64, 1)
    gradrec = True
    lr_dict = {0: 0.001, 50: 0.003}
    res = True
    cont = True
    torch.manual_seed(0)
    model = MODEL(res=res, cont=cont).to("cuda")
    modelname = 'HBNODE'
    print(model.__str__())
    rec = Recorder()
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[0])
    print('Number of Parameters: {}'.format(count_parameters(model)))
    timelist = [time.time()]
    for epoch in range(500):
        rec['epoch'] = epoch
        if epoch in lr_dict:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[epoch])

        batchsize = 256
        train_start_time = time.time()
        for b_n in range(0, len(data.train_x), batchsize):
            model.cell.nfe = 0
            predict = model(data.train_times[:, b_n:b_n + batchsize] / 64, data.train_x[:, b_n:b_n + batchsize])
            loss = criteria(predict, data.train_y[:, b_n:b_n + batchsize])
            rec['forward_nfe'] = model.cell.nfe
            rec['loss'] = loss.detach().cpu().numpy()



            # Gradient backprop computation
            if gradrec is not None:
                lossf = criteria(predict[-1], data.train_y[-1, b_n:b_n + batchsize])
                lossf.backward(retain_graph=True)
                vals = model.ode_rnn.h_rnn
                for i in range(len(vals)):
                    grad = vals[i].grad
                    rec['grad_{}'.format(i)] = 0 if grad is None else torch.norm(grad)
                model.zero_grad()

            

            model.cell.nfe = 0
            loss.backward()
            rec['backward_nfe'] = model.cell.nfe
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            wandb.log({"train_loss": loss.item()})
        rec['train_time'] = time.time() - train_start_time
        if epoch == 0 or (epoch + 1) % 1 == 0:
            model.cell.nfe = 0
            predict = model(data.valid_times / 64, data.valid_x)
            vloss = criteria(predict, data.valid_y)
            rec['va_nfe'] = model.cell.nfe
            rec['va_loss'] = vloss
        if epoch == 0 or (epoch + 1) % 20 == 0:
            model.cell.nfe = 0
            predict = model(data.test_times / 64, data.test_x)
            sloss = criteria(predict, data.test_y)
            sloss = sloss.detach().cpu().numpy()
            rec['ts_nfe'] = model.cell.nfe
            rec['ts_loss'] = sloss
        rec.capture(verbose=True)
        if (epoch + 1) % 20 == 0:
            torch.save(model, 'output/seq_{}_rnn_{}.mdl'.format(modelname, count_parameters(model)))
            rec.writecsv('output/seq_{}_rnn_{}.csv'.format(modelname, count_parameters(model)))
        wandb.log({"epoch": epoch, "train_loss": loss.item(), "va_loss": vloss.item(), "ts_loss": sloss})

if __name__ == "__main__":
    main()

"""


class NeuralODETimeSeries(pl.LightningModule):
    def __init__(self):
        super(NeuralODETimeSeries, self).__init__()
        self.odernn = ODE_RNN()  # Instantiate your ODE function

    def forward(self, x, t_span):
        z = self.ode(x, t_span)
        print("Z issssssssssssssssssss:::")
        print (type(z))
        print(z)
        return z

    def training_step(self, batch, batch_idx):

        t_span = torch.linspace(0, 1, len(batch))

        x, y = batch

        t_eval, y_hat = self(x, t_span)

        print("Type:t_eval")
        print(type(t_eval))
        print(t_eval)
        print("Type:y_hat")
        print(type(y_hat))
        print(y_hat)

        loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        dataset = Sequential_data(sample_num=1)[0]
        
        return DataLoader(dataset, num_workers=0, batch_size=10, shuffle=False)


def main():

    # Create a PyTorch Lightning model
    model = NeuralODETimeSeries(num_layers=3, hidden_size=128)

    # Create a PyTorch Lightning Trainer and train the model
    trainer = pl.Trainer(max_epochs=100, gpus=-1)
    trainer.fit(model)

if __name__ == "__main__":
    main()





# Create some data
batch, length, input_channels = 1, 1000, 258*128
hidden_channels = 256
t = torch.linspace(0, 1, length)
t_ = t.unsqueeze(0).unsqueeze(-1).expand(batch, length, 1)
x_ = Sequential_data(sample_num=1)
x_ = x_.unsqueeze(-1)
x = torch.cat([t_, x_], dim=2)  # include time as a channel

# Interpolate it
coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
X = torchcde.CubicSpline(coeffs)

# Create the Neural CDE system
class F(torch.nn.Module):
    def __init__(self):
        super(F, self).__init__()
        self.linear = torch.nn.Linear(hidden_channels,
                                      hidden_channels * input_channels)

    def forward(self, t, z):
        return self.linear(z).view(batch, hidden_channels, input_channels)

func = F()
z0 = torch.rand(batch, hidden_channels)

# Integrate it
torchcde.cdeint(X=X, func=func, z0=z0, t=X.interval)




class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y


######################
# Now we need some data.
# Here we have a simple example which generates some spirals, some going clockwise, some going anticlockwise.
######################
def get_data(num_timepoints=100):
    t = torch.linspace(0., 4 * math.pi, num_timepoints)

    start = torch.rand(128) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    ######################
    # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
    # rate at which time passes. Here, we have a regularly sampled dataset, so appending time is pretty simple.
    ######################
    X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos, y_pos], dim=2)
    y = torch.zeros(128)
    y[:64] = 1

    perm = torch.randperm(128)
    X = X[perm]
    y = y[perm]

    ######################
    # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)
    # y is a tensor of labels, of shape (batch=128,), either 0 or 1 corresponding to anticlockwise or clockwise
    # respectively.
    ######################
    return X, y


def main(num_epochs=30):
    train_X, train_y = get_data()

    ######################
    # input_channels=3 because we have both the horizontal and vertical position of a point in the spiral, and time.
    # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
    # output_channels=1 because we're doing binary classification.
    ######################
    model = NeuralCDE(input_channels=3, hidden_channels=8, output_channels=1)
    optimizer = torch.optim.Adam(model.parameters())

    ######################
    # Now we turn our dataset into a continuous path. We do this here via Hermite cubic spline interpolation.
    # The resulting `train_coeffs` is a tensor describing the path.
    # For most problems, it's probably easiest to save this tensor and treat it as the dataset.
    ######################
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    test_X, test_y = get_data()
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
    pred_y = model(test_coeffs).squeeze(-1)
    binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
    prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
    proportion_correct = prediction_matches.sum() / test_y.size(0)
    print('Test Accuracy: {}'.format(proportion_correct))


if __name__ == '__main__':
    main()





# something idk


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
        #self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128,128)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, t, x):
        # Implement the dynamics of your system here
        # You can use the defined layers and return dx/dt
        #out = self.relu(self.conv1(x))
        #out = self.relu(self.conv2(out))
        out = self.relu1(self.lin1(x))
        out = self.relu2(self.lin2(out))
        return out

# In your NeuralODETimeSeries class
class NeuralODETimeSeries(pl.LightningModule):
    def __init__(self, num_layers, hidden_size):
        super(NeuralODETimeSeries, self).__init__()
        self.func = ODEFunc()  # Instantiate your ODE function
        self.ode = NeuralODE(self.func, sensitivity='autograd', solver='dopri5')
        #self.fc = torch.nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x, t_span):
        z = self.ode(x, t_span)
        print("Z issssssssssssssssssss:::")
        print (type(z))
        print(z)
        return z

    def training_step(self, batch, batch_idx):

        t_span = torch.linspace(0, 1, len(batch))

        x, y = batch

        t_eval, y_hat = self(x, t_span)

        print("Type:t_eval")
        print(type(t_eval))
        print(t_eval)
        print("Type:y_hat")
        print(type(y_hat))
        print(y_hat)

        loss = torch.nn.functional.mse_loss(y_hat, y)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        dataset = Sequential_data(sample_num=1)[0]
        
        return DataLoader(dataset, num_workers=0, batch_size=10, shuffle=False)


def main():

    # Create a PyTorch Lightning model
    model = NeuralODETimeSeries(num_layers=3, hidden_size=128)

    # Create a PyTorch Lightning Trainer and train the model
    trainer = pl.Trainer(max_epochs=100, gpus=-1)
    trainer.fit(model)

if __name__ == "__main__":
    main()


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