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
from matcho import Unet2D
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
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Implement your custom loss calculation here
        # loss = torch.mean((y_pred - y_true) ** 2)  # Example: Mean Squared Error
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


def preprocess(traj, Par):
    x = sliding_window_view(traj[:,:-(Par['lf']-1),:,:], window_shape=Par['lb'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lb'],Par['nx'], Par['ny'])
    y = sliding_window_view(traj[:,Par['lb']-1:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,4,2,3).reshape(-1,Par['lf'],Par['nx'], Par['ny'])
    t = np.linspace(0,1,Par['lf']).reshape(-1,1)

    nt = y.shape[1]
    n_samples = y.shape[0]

    t = np.tile(t, [n_samples,1]).reshape(-1,)                                                     #[_*nt, ]
    x = np.repeat(x,nt, axis=0)                                   #[_*nt, 1, 64, 64]
    y = y.reshape(y.shape[0]*y.shape[1],1,y.shape[2],y.shape[3])  #[_*nt, 64, 64]


    print('x: ', x.shape)
    print('y: ', y.shape)
    print('t: ', t.shape)
    print()
    return x,y,t


# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################
res = 128
begin_time = time.time()
traj = np.load(f"/oscar/data/gk/voommen/no_diffusion/kolmogrov/data/alpha_1.5_tau_14_re_2007_N_1000_T_50_nt_200_nx_512/res_{res}/traj.npy") #[1000, nx, ny, nt]
traj = traj.transpose(0,3,1,2)
print(f"Data Loading Time: {time.time() - begin_time:.1f}s")

traj_train = traj[:800, :80][:, ::2]
traj_val   = traj[800:900, :80][:, ::2]
traj_test  = traj[900:, :80][:, ::2]

print(f"traj_train: {traj_train.shape}")
print(f"traj_val: {traj_val.shape}")
print(f"traj_test: {traj_test.shape}")

Par = {}
# Par['nt'] = 100 
Par['nx'] = traj_train.shape[2]
Par['ny'] = traj_train.shape[3]
Par['nf'] = 1
Par['d_emb'] = 128

Par['lb'] = 20
Par['lf'] = 20+1 
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

train_dataset = YourDataset(x_train_tensor, t_train_tensor, y_train_tensor)
val_dataset = YourDataset(x_val_tensor, t_val_tensor, y_val_tensor)
test_dataset = YourDataset(x_test_tensor, t_test_tensor, y_test_tensor)

# Define data loaders
train_batch_size = 100
val_batch_size   = 100
test_batch_size  = 100
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Initialize your Unet2D model
if res==64:
    model = Unet2D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8)).to(device).to(torch.float32)
elif res==128:
    model = Unet2D(dim=16, channels=Par['lb'], Par=Par, dim_mults=(1, 2, 4, 8)).to(device).to(torch.float32)
summary(model, input_size=((1,)+x_train.shape[1:], (1,)) )

# Adjust the dimensions as per your model's input size
dummy_x = torch.tensor(torch.randn(1, 20, 128,128),   dtype=DTYPE, device=device)
dummy_t = torch.tensor(torch.randn(1,),   dtype=DTYPE, device=device)
dummy_input = (dummy_x, dummy_t)

# Profile the model
flops = torchprofile.profile_macs(model, dummy_input)
print(f"FLOPs: {flops:.2e}")

# Define loss function and optimizer
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

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

    for x, t, y_true in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()
        with autocast():
            y_pred = model(x.to(device), t.to(device))
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
        for x, t, y_true in val_loader:
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
    for x, t, y_true in test_loader:
        with autocast():
            y_pred = model(x.to(device), t.to(device))
            loss = criterion(y_pred, y_true.to(device))
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')

