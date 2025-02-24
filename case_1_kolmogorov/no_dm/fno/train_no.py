import os
import sys

import math
import time
import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import torchprofile

from neuralop.models import TFNO3d

import pickle

torch.manual_seed(23)

scaler = GradScaler()

DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define your custom loss function here
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Implement your custom loss calculation here
        # loss = torch.mean((y_pred - y_true) ** 2)  # Example: Mean Squared Error
        loss = torch.norm(y_true-y_pred, p=2)/torch.norm(y_true, p=2)
        return loss

class YourDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)

        return x_sample, y_sample
    
class Normalizer():
    def __init__(self, shift, scale):
        self.shift = torch.tensor(shift, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
    
    def normalize(self, x):
        return (x - self.shift)/self.scale
    
    def renormalize(self, x):
        return x*self.scale + self.shift

def get_xyt_grid(nx=None, ny=None, nt=None, bot=[0, 0, 0], top=[1, 1, 1], dtype='float32',
                 x_arr=None, y_arr=None, t_arr=None, dt=0.):
    '''
    Args:
        S: number of points on each spatial domain
        T: number of points on temporal domain including endpoint
        bot: list or tuple, lower bound on each dimension
        top: list or tuple, upper bound on each dimension

    Returns:
        (n_x, n_y, n_t, 3) array of grid points
    '''
    if x_arr is None:
        x_arr = np.linspace(bot[0], top[0], num=nx, endpoint=True)

    if y_arr is None:
        y_arr = np.linspace(bot[1], top[1], num=ny, endpoint=True)

    if t_arr is None:
        if dt is None:
            dt = (top[2] - bot[2]) / nt
        t_arr = np.linspace(bot[2] + dt, top[2], num=nt)

    x_grid, y_grid, t_grid = np.meshgrid(x_arr, y_arr, t_arr, indexing='ij')
    x_axis = np.ravel(x_grid)
    y_axis = np.ravel(y_grid)
    t_axis = np.ravel(t_grid)
    grid = np.stack([x_axis, y_axis, t_axis], axis=0).T

    x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')
    x_axis = np.ravel(x_grid)
    y_axis = np.ravel(y_grid)
    grid_space = np.stack([x_axis, y_axis], axis=0).T

    grids_dict = {'grid_x': x_arr, 
                  'grid_y': y_arr,
                  'grid_t': t_arr, 
                  'grid_space': grid_space, 
                  'grid': grid}
    for key in grids_dict.keys():
        grids_dict[key] = torch.tensor(grids_dict[key], dtype=eval('torch.' + dtype), device='cpu')
    return grids_dict



def _preprocess_fno(data_dict, Par):
    Grid_train = get_xyt_grid(Par['nx'], Par['ny'], Par['lf'], bot=[0, 0, 0], top=[1, 1, 1], dtype='float32')
    Grid_test  = get_xyt_grid(Par['nx'], Par['ny'], Par['lf'], bot=[0, 0, 0], top=[1, 1, 1], dtype='float32')
    
    data_dim = len(data_dict['x_train'].shape) - 2
    if data_dim != len(data_dict['x_test'].shape) - 2:
        raise ValueError('Data dimension mismatch between train and test sets.' +
                            f'Train: {data_dim}, Test: {len(data_dict["x_test"].shape)}')

    # Repeat shape is [1, time_steps_train, 1, 1, 1] for 3D data
    time_steps_train = Par['lf'] #self.configs.time_steps_train
    time_steps_val = Par['lf'] #self.configs.time_steps_inference if self.configs.scenario == 'hypersonics' \
    time_steps_test = Par['lf'] #self.configs.time_steps_inference  

    for dataset in ['x_train', 'x_val', 'x_test']:
        n_samples = data_dict[dataset].shape[0]
        if True:
            grid = Grid_test if dataset == 'x_test' else Grid_train
            time_steps = time_steps_test if dataset == 'x_test' else time_steps_train

        grid_t = grid['grid_t'].reshape([1, time_steps] + [1] * data_dim)
        grid_t = grid_t.repeat(
            [n_samples, 1] + list(data_dict[dataset].shape[2:]))
        if data_dim == 1:
            x_shape = [1, 1] + list(data_dict[dataset].shape[2:3])
            grid_x = grid['grid_x'].reshape(
                x_shape).repeat(n_samples, time_steps, 1)
            data_dict[dataset] = torch.stack(
                [data_dict[dataset], grid_x, grid_t], dim=1)
        elif data_dim == 2:
            x_shape = [1, 1] + list(data_dict[dataset].shape[2:3]) + [1]
            y_shape = [1, 1, 1] + list(data_dict[dataset].shape[3:])

            grid_x = grid['grid_x'].reshape(x_shape).repeat(
                n_samples, time_steps, 1, y_shape[-1])
            grid_y = grid['grid_y'].reshape(y_shape).repeat(
                n_samples, time_steps, x_shape[-2], 1)

            data_dict[dataset] = torch.stack(
                [data_dict[dataset], grid_x, grid_y, grid_t], dim=1)
        elif data_dim == 3:
            x_shape = [1, 1] + list(data_dict[dataset].shape[2:3]) + [1, 1]
            y_shape = [1, 1, 1] + list(data_dict[dataset].shape[3:4]) + [1]
            z_shape = [1, 1, 1, 1] + list(data_dict[dataset].shape[4:])
            grid_x = grid['grid_x'].reshape(x_shape)
            grid_y = grid['grid_y'].reshape(y_shape)
            grid_z = grid['grid_z'].reshape(z_shape)

            grid_x = grid_x.repeat(
                n_samples, time_steps, 1, y_shape[-2], z_shape[-1])
            grid_y = grid_y.repeat(
                n_samples, time_steps, x_shape[-3], 1, z_shape[-1])
            grid_z = grid_z.repeat(
                n_samples, time_steps, x_shape[-3], y_shape[-2], 1)

            data_dict[dataset] = torch.stack(
                [data_dict[dataset], grid_x, grid_y, grid_z, grid_t], dim=1)
    return data_dict



def preprocess(traj, Par):
    x = sliding_window_view(traj[:,:-(Par['lf']),:,:], window_shape=Par['lb'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lb'],Par['nx'], Par['ny'])
    y = sliding_window_view(traj[:,Par['lb']:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])
    t = np.linspace(0,1,Par['lf']).reshape(-1,1)

    nt = y.shape[1]
    n_samples = y.shape[0]

    # t = np.tile(t, [n_samples,1]).reshape(-1,)                                                     #[_*nt, ]
    # x = np.repeat(x,nt, axis=0)                                   #[_*nt, 1, 64, 64]
    # y = y.reshape(y.shape[0]*y.shape[1],1,y.shape[2],y.shape[3])  #[_*nt, 64, 64]


    print('x: ', x.shape)
    print('y: ', y.shape)
    print('t: ', t.shape)
    print()
    return x,y,t


# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################
res = 128
begin_time = time.time()

# Load dataset from the correct path
traj = np.load(f"/oscar/data/gk/voommen/no_diffusion/kolmogrov/data/alpha_1.5_tau_14_re_2007_N_1000_T_50_nt_200_nx_512/res_{res}/traj.npy") #[1000, nx, ny, nt]
traj = traj.transpose(0,3,1,2)
print(f"Data Loading Time: {time.time() - begin_time:.1f}s")


# each simulation, t \in [0s, 50s] in 201 timesteps

traj_train = traj[:800, :80][:, ::2]     #t \in [0s, 20s] in 40 steps
traj_val   = traj[800:900, :80][:, ::2]
traj_test  = traj[900:, :80][:, ::2]

Par = {}
# Par['nt'] = 100 
Par['nx'] = traj_train.shape[2]
Par['ny'] = traj_train.shape[3]
Par['nf'] = 1
Par['d_emb'] = 128

Par['lb'] = 20
Par['lf'] = 20 
# Par['temp'] = Par['nt'] - Par['lb'] - Par['lf'] + 2

Par['num_epochs'] = 500

begin_time = time.time()
print('\nTrain Dataset')
x_train, y_train, t_train = preprocess(traj_train, Par)
print('\nValidation Dataset')
x_val, y_val, t_val  = preprocess(traj_val, Par)
print('\nTest Dataset')
x_test, y_test, t_test  = preprocess(traj_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

t_min = np.min(t_train)
t_max = np.max(t_train)

Par['inp_shift'] = np.mean(x_train) 
Par['inp_scale'] = np.std(x_train)
Par['out_shift'] = np.mean(y_train)
Par['out_scale'] = np.std(y_train)
Par['t_shift']   = t_min
Par['t_scale']   = t_max - t_min

inp_normalizer = Normalizer(Par['inp_shift'], Par['inp_scale'])
out_normalizer = Normalizer(Par['out_shift'], Par['out_scale'])

with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

# sys.exit()
#########################

# Create custom datasets
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
t_train_tensor = torch.tensor(t_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_val_tensor   = torch.tensor(x_val,   dtype=torch.float32)
t_val_tensor   = torch.tensor(t_val,   dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val,   dtype=torch.float32)

x_test_tensor  = torch.tensor(x_test,  dtype=torch.float32)
t_test_tensor  = torch.tensor(t_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

data_dict = {'x_train':x_train_tensor, 'x_val':x_val_tensor, 'x_test':x_test_tensor}
data_dict = _preprocess_fno(data_dict, Par)

x_train_tensor = torch.tensor(data_dict['x_train'].cpu().numpy(), dtype=torch.float32)
x_val_tensor   = torch.tensor(data_dict['x_val'].cpu().numpy()  , dtype=torch.float32)
x_test_tensor  = torch.tensor(data_dict['x_test'].cpu().numpy() , dtype=torch.float32)

print(f"x_train_tensor: {x_train_tensor.shape}")
print(f"x_val_tensor: {x_val_tensor.shape}")
print(f"x_test_tensor: {x_test_tensor.shape}")

train_dataset = YourDataset(x_train_tensor, y_train_tensor)
val_dataset = YourDataset(x_val_tensor, y_val_tensor)
test_dataset = YourDataset(x_test_tensor, y_test_tensor)

# Define data loaders
train_batch_size = 10
val_batch_size   = 10
test_batch_size  = 10
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Initialize your Unet2D model
model = TFNO3d(12, 12, 12, hidden_channels=20, in_channels=4, out_channels=1).cuda()
summary(model, input_size=((1,)+x_train_tensor.shape[1:]) )

# Adjust the dimensions as per your model's input size
dummy_x = x_train_tensor[0:1]
dummy_input = dummy_x.to(device)

# Profile the model
flops = torchprofile.profile_macs(model, dummy_input)
print(f"FLOPs: {flops:.2e}")

# Define loss function and optimizer
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed

# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0

os.makedirs('models', exist_ok=True)
t0 = time.time()
for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0

    for x, y_true in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        with autocast():
            x = inp_normalizer.normalize(x)
            y_pred = out_normalizer.renormalize(model(x.to(device))[:,0]) 
            loss   = criterion(y_pred, y_true.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # Update learning rate
        scheduler.step()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y_true in val_loader:
            with autocast():
                x = inp_normalizer.normalize(x)
                y_pred = out_normalizer.renormalize(model(x.to(device))[:,0]) 
                loss   = criterion(y_pred, y_true.to(device))
            val_loss += loss.item()

    val_loss /= len(val_loader)

        # Save the model if validation loss is the lowest so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        torch.save(model.state_dict(), f'models/best_model.pt')
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}')

print('Training finished.')
print(f"Training Time: {time.time() - t0:.1f}s")

# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x, y_true in test_loader:
        with autocast():
            x = inp_normalizer.normalize(x)
            y_pred = out_normalizer.renormalize(model(x.to(device))[:,0]) 
            loss = criterion(y_pred, y_true.to(device))
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')

