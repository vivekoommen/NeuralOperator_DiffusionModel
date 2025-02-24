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
from MnM import MnM
from torchinfo import summary
import torchprofile

# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import pickle
torch.manual_seed(23)

DTYPE = torch.float32

scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class CustomLoss(nn.Module):
    def __init__(self, Par):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true, Par):
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


def preprocess_train(traj, Par):
    # traj - [bs, nt, nx, ny, nf]

    subsample_t = Par['subsample_t']
    x = sliding_window_view(traj[:,:-(Par['lf']),:,:], window_shape=Par['lb'], axis=1 ).transpose(0,1,5,2,3,4)[:,::subsample_t].reshape(-1,Par['lb'],Par['nx'], Par['ny'],Par['nf'])
    y = sliding_window_view(traj[:,Par['lb']:,:,:], window_shape=Par['lf'], axis=1 ).transpose(0,1,5,2,3,4)[:,::subsample_t].reshape(-1,Par['lf'],Par['nx'], Par['ny'],Par['nf'])

    x = x.transpose(0,4,1,2,3) #[bs, nf, lb,nx,ny]
    y = y.transpose(0,4,1,2,3) #[bs, nf, lf,nx,ny]

    print('x: ', x.shape)
    print('y: ', y.shape)
    print()
    return x.astype(np.float32), y.astype(np.float32)


def preprocess(traj, Par):
    # traj - [bs , nt, nx , ny, nf]
    x = traj[:,:Par['lb']  ] #[bs, lb, nx,ny,nf]
    y = traj[:, Par['lb']: ] #[bs , nt-lb, nx ,ny, nf]

    x = x.transpose(0,4,1,2,3) #[bs, nf, lb,nx,ny]
    y = y.transpose(0,4,1,2,3) #[bs, nf, nt-lb,nx,ny]

    print('x: ', x.shape)
    print('y: ', y.shape)
    print()

    return x.astype(np.float32), y.astype(np.float32)

def rollout(model, x, bs, Par):
    # x - [bs, nf, lb,nx,ny]

    NT = Par['nt']
    y_pred_ls = []
    # lf = 
    # nx = 64
    # ny = 64

    bs = bs
    end= bs
    for end in range(bs, x.shape[0]+1, bs):
        # print('end: ', end)
        start = end-bs
        out_ls = [x[start:end, :, :Par['lb']].to(device)]
        
        temp_x1 = x[start:end, :, -Par['lb']:] #[BS, nf, lb, nx,ny]
        while (len(out_ls)-1)*(Par['lf'])<NT:
            model.eval()
            with torch.no_grad():
                temp_x = temp_x1
                # print('temp_x: ', temp_x.shape)
                with autocast():
                    out = model(temp_x.to(device)) #[BS, nf, lf,nx,ny]
                out_ls.append(out.to(device))
                temp_out = torch.cat(out_ls, dim=2)
                temp_x1 = temp_out[:,:,-Par['lb']:] #[BS, nf, lb,nx,ny]
                                
        pred = torch.cat(out_ls, dim=2)[:, :, Par['lb']:NT] #[BS, nf, nt-lb, nx, ny]
        # print('pred: ', pred.shape)
        y_pred_ls.append(pred)

    y_pred = torch.cat(y_pred_ls, dim=0)#.reshape(1,-1,Par['nz'],Par['ny'],Par['nx'])
    # print('y_pred: ', y_pred.shape)

    return y_pred

def combined_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################


res=128
print("Loading Dataset ...")
debug = False

traj = np.load(f"/oscar/data/gk/voommen/no_diffusion/kolmogrov/data/alpha_1.5_tau_14_re_2007_N_1000_T_50_nt_200_nx_512/res_{res}/traj.npy") #[1000, nx, ny, nt]
traj = traj.transpose(0,3,1,2)
traj = np.expand_dims(traj, axis=-1)

traj_train = traj[:800, :80][:, ::2]
traj_val   = traj[800:900, :80][:, ::2]
traj_test  = traj[900:, :80][:, ::2]

print("Loaded Dataset")
print("Dataset type: ", traj_train.dtype)



MEAN = np.mean(traj_train).reshape(1,-1,1,1,1)
STD  = np.std(traj_train).reshape(1,-1,1,1,1)
MIN  = np.min(traj_train).reshape(1,-1,1,1,1)
MAX  = np.max(traj_train).reshape(1,-1,1,1,1)

print(f"MEAN: {MEAN.shape}\nSTD: {STD.shape}\nMIN: {MIN.shape}\nMAX: {MAX.shape}")

'''
change n_channels
'''

Par = {
       'DEVICE'          : device,
       'nt'              : traj_train.shape[1],
       'nx'              : traj_train.shape[2],
       'ny'              : traj_train.shape[3],
       'nf'              : traj_train.shape[4],
       'lb'              : 20,
       'lf'              : 10,
       'subsample_t'     : 1
       }

print('\nTrain Dataset')
x_train,y_train = preprocess_train(traj_train, Par)
print('\nValidation Dataset')
x_val,y_val = preprocess(traj_val, Par)
print('\nTest Dataset')
x_test,y_test = preprocess(traj_test, Par)

# sys.exit()

Par.update(
       {
       'inp_ch'          : Par['nf']*Par['lb'],
       'out_ch'          : Par['nf']*Par['lf'],
       'n_channels'      : 16,
       'k'               : 3,
       'block_type_ls'  : ['Conv'],
       'inp_shift'       : torch.tensor(MEAN, dtype=DTYPE, device=device),
       'inp_scale'       : torch.tensor(STD, dtype=DTYPE, device=device),
       'out_shift'       : torch.tensor(MEAN, dtype=DTYPE, device=device),
       'out_scale'       : torch.tensor(STD, dtype=DTYPE, device=device),
       'out_shift_loss'  : 0,
       'out_scale_loss'  : 1,
       }
)


if debug:
    Par['num_epochs']  = 50 #500 #500
else:
    Par['num_epochs']  = 500 #4000

print('Par:\n', Par)

with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

# sys.exit()
#########################


# Create custom datasets
x_train_tensor = torch.tensor(x_train, dtype=DTYPE)
y_train_tensor = torch.tensor(y_train, dtype=DTYPE)

x_val_tensor   = torch.tensor(x_val,   dtype=DTYPE)
y_val_tensor   = torch.tensor(y_val,   dtype=DTYPE)

x_test_tensor  = torch.tensor(x_test,  dtype=DTYPE)
y_test_tensor  = torch.tensor(y_test,  dtype=DTYPE)

train_dataset = YourDataset(x_train_tensor, y_train_tensor)
val_dataset = YourDataset(x_val_tensor, y_val_tensor)
test_dataset = YourDataset(x_test_tensor, y_test_tensor)

# Define data loaders
train_batch_size = 32
val_batch_size   = x_val.shape[0]
test_batch_size  = x_test.shape[0]
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Initialize your Unet2D model
model = MnM(Par).to(device).to(DTYPE)
summary(model, input_size=(1,)+x_train.shape[1:])

# Adjust the dimensions as per your model's input size
dummy_input = torch.tensor(torch.randn(1, Par['nf'], Par['lb'], Par['nx'],Par['ny']),   dtype=DTYPE, device=device)

# Profile the model
flops = torchprofile.profile_macs(model, dummy_input)
print(f"FLOPs: {flops:.2e}")

# Define loss function and optimizer
criterion = CustomLoss(Par)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed
# scheduler = combined_scheduler(optimizer, Par['num_epochs'] * len(train_loader), int(0.1 * Par['num_epochs']) * len(train_loader))


# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0

os.makedirs('models', exist_ok=True)

for epoch in range(num_epochs):
    begin_time = time.time()
    model.train()
    train_loss = 0.0
    counter=0

    for x, y_true in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        
        # for param in model.parameters():
        #     print("w: ", param.device.type)

        optimizer.zero_grad()
        # with autocast():
        if True:
            y_pred = model(x.to(device))
            loss   = criterion(y_pred, y_true.to(device), Par)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        counter += 1

        # Update learning rate
        scheduler.step()

    train_loss /= counter

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y_true in val_loader:
            with autocast():
                y_pred = rollout(model, x, val_batch_size, Par)
                loss = criterion(y_pred, y_true.to(device), Par).item() 
            val_loss += loss

    val_loss /= len(val_loader)

     # Save the model if validation loss is the lowest so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_id = epoch+1
        torch.save(model.state_dict(), f'models/best_model.pt')
    
    time_stamp = str('[')+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+str(']')
    elapsed_time = time.time() - begin_time
    # print(f'Mean sigma1j,j: {div_1:.4e} '+ f'Mean sigma2j,j: {div_2:.4e} '+ f'Mean sigma3j,j: {div_3:.4e} ')
    print(time_stamp + f' - Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, best model: {best_model_id}, LR: {scheduler.get_last_lr()[0]:.4e}, epoch time: {elapsed_time:.2f}'
          )

print('Training finished.')

# Testing loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x, y_true in test_loader:
        with autocast():
            y_pred = rollout(model, x, test_batch_size, Par)
            loss = criterion(y_pred, y_true.to(device), Par).item() 
        test_loss += loss 
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')
