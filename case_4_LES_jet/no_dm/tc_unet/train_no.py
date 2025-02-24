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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from matcho_3d import Unet3D
# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import torchprofile

import pickle

torch.manual_seed(23)

scaler = GradScaler()

DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define your custom loss function here
class CustomLoss(nn.Module):
    def __init__(self, Par):
        super(CustomLoss, self).__init__()
        self.Par = Par

    def forward(self, y_pred, y_true):
        # Implement your custom loss calculation here
        # loss = torch.mean((y_pred - y_true) ** 2)  # Example: Mean Squared Error
        y_true = (y_true - self.Par["out_shift"])/self.Par["out_scale"]
        y_pred = (y_pred - self.Par["out_shift"])/self.Par["out_scale"]
        loss = torch.norm(y_true-y_pred, p=2)/torch.norm(y_true, p=2)
        return loss

class YourDataset(Dataset):
    def __init__(self, x, t, y, transform=None):
        self.x = x
        self.t = t
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        t_sample = self.t[idx]
        y_sample = self.y[idx]

        if self.transform:
            x_sample, t_sample, y_sample = self.transform(x_sample, t_sample, y_sample)

        return x_sample, t_sample, y_sample


# def preprocess(traj, Par):
#     x = sliding_window_view(traj[:,:-(Par['lf']),:,:,:,:], window_shape=Par['lb'], axis=1 ).transpose(0,1,6,2,3,4,5).reshape(-1,Par['lb'], Par['nf'], Par['nth'], Par['nr'], Par['nx'])
#     y = sliding_window_view(traj[:,Par['lb']:,:,:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,6,2,3,4,5).reshape(-1,Par['lf'], Par['nf'], Par['nth'], Par['nr'], Par['nx'])
#     t = np.linspace(0,1,Par['lf']).reshape(-1,1)

#     nt = y.shape[1]
#     n_samples = y.shape[0]

#     t = np.tile(t, [n_samples,1]).reshape(-1,)                                                     #[_*nt, ]
#     x = np.repeat(x,nt, axis=0)                                   #[_*nt, 1, 64, 64]
#     y = y.reshape(y.shape[0]*y.shape[1],1,y.shape[2],y.shape[3])  #[_*nt, 64, 64]


#     print('x: ', x.shape)
#     print('y: ', y.shape)
#     print('t: ', t.shape)
#     print()
#     return x,y,t

def preprocess(traj, Par):
    nsamples = traj.shape[0]
    nt = traj.shape[1]
    temp = nt - Par['lb'] - Par['lf'] + 1
    x_idx = np.arange(temp).reshape(-1,1)
    x_idx = np.tile(x_idx, (1, Par['lf'])).reshape(-1,)

    t_idx = np.arange(Par['lf']).reshape(1,-1)
    t_idx = np.tile(t_idx, (temp,1)).reshape(-1,)

    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par['lb']:], window_shape=Par['lf']).reshape(-1,)

    print(f"x_idx: {x_idx.shape}")
    print(f"t_idx: {t_idx.shape}")
    print(f"y_idx: {y_idx.shape}")

    return x_idx, t_idx, y_idx


def combined_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################
debug = False

res = 128
begin_time = time.time()
if debug:
    traj = np.load(f"../data/velocity_sample.npy")[:,:,::2, ::2, ::2] #[nt, nf, nth, nr, nx]
else:
    traj = np.load("../data/velocity_vec_3d.npy")[:,:,::2, ::2, ::2]

traj = np.expand_dims(traj, axis=0) #[1, nt, nf, nth, nr, nx]
print(f"Data Loading Time: {time.time() - begin_time:.1f}s")

print(f"traj: {traj.shape}")

idx1 = int(0.8 * traj.shape[1])
idx2 = int(0.9 * traj.shape[1])

traj_train = traj[:, :idx1]
traj_val   = traj[:, idx1:idx2]
traj_test  = traj[:, idx2:]


Par = {}
# Par['nt'] = 100 
Par['nf'] = traj_train.shape[2]
Par['nth'] = traj_train.shape[3]
Par['nr'] = traj_train.shape[4]
Par['nx'] = traj_train.shape[5]
Par['d_emb'] = 128

Par['lb'] = 1
Par['lf'] = 10
Par['channels'] = Par['nf']
# Par['temp'] = Par['nt'] - Par['lb'] - Par['lf'] + 2

Par['num_epochs'] = 50

time_cond = np.linspace(0, 1, Par['lf'])

begin_time = time.time()
print('\nTrain Dataset')
x_idx_train, t_idx_train, y_idx_train = preprocess(traj_train, Par)
print('\nValidation Dataset')
x_idx_val, t_idx_val, y_idx_val  = preprocess(traj_val, Par)
print('\nTest Dataset')
x_idx_test, t_idx_test, y_idx_test  = preprocess(traj_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

# sys.exit()

t_min = np.min(time_cond)
t_max = np.max(time_cond)

MEAN = np.load("../data/MEAN_vel.npy").reshape(1,-1,1,1,1)
STD  = np.load("../data/STD_vel.npy").reshape(1,-1,1,1,1)
MIN  = np.load("../data/MIN_vel.npy").reshape(1,-1,1,1,1)
MAX  = np.load("../data/MAX_vel.npy").reshape(1,-1,1,1,1)

Par['inp_shift'] = torch.tensor(MEAN, dtype=DTYPE, device=device)
Par['inp_scale'] = torch.tensor(STD, dtype=DTYPE, device=device)
Par['out_shift'] = torch.tensor(MEAN, dtype=DTYPE, device=device)
Par['out_scale'] = torch.tensor(STD, dtype=DTYPE, device=device)
Par['t_shift']   = t_min
Par['t_scale']   = t_max - t_min

with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

# sys.exit()
#########################

# Create custom datasets
traj_train_tensor = torch.tensor(traj_train, dtype=DTYPE)
traj_val_tensor = torch.tensor(traj_val, dtype=DTYPE)
traj_test_tensor = torch.tensor(traj_test, dtype=DTYPE)
time_cond_tensor = torch.tensor(time_cond, dtype=DTYPE)


train_dataset = YourDataset(x_idx_train, t_idx_train, y_idx_train)
val_dataset = YourDataset(x_idx_val, t_idx_val, y_idx_val)
test_dataset = YourDataset(x_idx_test, t_idx_test, y_idx_test)

# Define data loaders
train_batch_size = 10
val_batch_size   = 10
test_batch_size  = 10
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Initialize your Unet3D model
model = Unet3D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8), channels=Par['channels']).to(device).to(torch.float32)
summary(model, input_size=((1, 3, 32, 32, 128), (1,)) )

# Adjust the dimensions as per your model's input size
dummy_x = traj_train_tensor[0,[0]].to(device)
dummy_t = time_cond_tensor[0:1].to(device)
dummy_input = (dummy_x, dummy_t)

# Profile the model
flops = torchprofile.profile_macs(model, dummy_input)
print(f"FLOPs: {flops:.2e}")

# Define loss function and optimizer
criterion = CustomLoss(Par)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Learning rate scheduler (Cosine Annealing)
# scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed
scheduler = combined_scheduler(optimizer, Par['num_epochs'] * len(train_loader), int(0.1 * Par['num_epochs']) * len(train_loader))


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

    for x_idx, t_idx, y_idx in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        x = traj_train_tensor[0, x_idx].to(device)
        t = time_cond_tensor[t_idx].to(device)
        y_true = traj_train_tensor[0, y_idx].to(device)
        optimizer.zero_grad()
        with autocast():
            y_pred = model(x, t)
            loss   = criterion(y_pred, y_true)
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
        for x_idx, t_idx, y_idx in val_loader:
            x = traj_val_tensor[0, x_idx]
            t = time_cond_tensor[t_idx]
            y_true = traj_val_tensor[0, y_idx]
            with autocast():
                y_pred = model(x.to(device), t.to(device))
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
    for x_idx, t_idx, y_idx in test_loader:
        x = traj_test_tensor[0, x_idx]
        t = time_cond_tensor[t_idx]
        y_true = traj_test_tensor[0, y_idx]
        with autocast():
            y_pred = model(x.to(device), t.to(device))
            loss = criterion(y_pred, y_true.to(device))
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')

