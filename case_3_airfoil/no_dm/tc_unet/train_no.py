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
from matcho import Unet2D
# from YourDataset import YourDataset  # Import your custom dataset here
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
import torchprofile
import matplotlib.pyplot as plt
import scipy.stats as stats

import pickle

torch.manual_seed(23)

scaler = GradScaler()

DTYPE = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def make_plot(TRUE, PRED, epoch):
    sample_id = 0
    skip_t = 4
    t_ls = np.arange(11)[1:]
    true = TRUE[sample_id, ::skip_t, 0]#[1:]
    pred1 = PRED[sample_id, ::skip_t, 0]#[1:]
    time_ls = t_ls[::skip_t]#[1:]

    # print(true.shape)
    # print(time_ls)

    CMAP = 'turbo'

    # Function to calculate MSE
    def mse(a, b):
        return np.mean((a - b) ** 2)

    def rel_l2(T, P):
        T = torch.tensor(T, dtype=DTYPE)
        P = torch.tensor(P, dtype=DTYPE)
        return torch.norm(T-P, p=2)/torch.norm(T, p=2).cpu().numpy()

    # print(rel_l2(Y_TRUE[sample_id],Y_PRED[sample_id]))

    # Create the figure and axis
    fig, axes = plt.subplots(4, 3, figsize=(30, 11))
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # Colorbar axis

    # Find the min and max values for colorbar
    vmin = min(true.min(), pred1.min())
    vmax = max(true.max(), pred1.max())

    # Plot the images
    for i in range(3):
        # Ground truth row
        im = axes[0, i].imshow(true[i], vmin=vmin, vmax=vmax, cmap=CMAP)
        axes[0, i].set_title(f'Time: {time_ls[i]}s', fontsize=16)
        axes[0, i].axis('off')

        # Prediction 1 row
        im = axes[1, i].imshow(pred1[i], vmin=vmin, vmax=vmax, cmap=CMAP)
        mse_val1 = rel_l2(true[i], pred1[i])
        axes[1, i].set_title(f'rel L2: {mse_val1:.2e}', fontsize=12)
        axes[1, i].axis('off')



        #######################

        image = true[i]
        
        ny, nx = image.shape
        
        # Compute the Fourier transform and get the amplitude squared
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2
        
        # Create the k-frequency grid
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2)
        
        # Flatten the arrays to use in binning
        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        
        # Define the bins for the wavenumber
        kbins = np.arange(0.5, min(nx, ny)//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        
        # Bin the data
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                            statistic="mean",
                                            bins=kbins)
        
        # Scale the binned amplitudes
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        
        
        
        # Plotting
        
        axes[2, i].loglog(kvals, Abins, label="Simulated")

        image = pred1[i]
        
        ny, nx = image.shape
        
        # Compute the Fourier transform and get the amplitude squared
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2
        
        # Create the k-frequency grid
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2)
        
        # Flatten the arrays to use in binning
        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        
        # Define the bins for the wavenumber
        kbins = np.arange(0.5, min(nx, ny)//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        
        # Bin the data
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                            statistic="mean",
                                            bins=kbins)
        
        # Scale the binned amplitudes
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        
        
        
        # Plotting
        
        axes[2, i].loglog(kvals, Abins, label="MATCHO")

        # Adding the -5/3 slope line
        k_ref = np.linspace(1, np.max(kbins), 100)  # Range of wavenumber for the reference line
        energy_ref = k_ref**(-5/3)  # Energy corresponding to the -5/3 slope
        energy_ref *= max(Abins) / max(energy_ref)  # Normalize to plot scale
        if i != 0:
            axes[2, i].loglog(k_ref, energy_ref, 'k--', label='k^-5/3 Reference')


        axes[2, i].set_xlabel('$k$')
        if i == 5:
            axes[2, i].legend()

        ################## 

        image = true[i]
        
        ny, nx = image.shape
        
        # Compute the Fourier transform and get the amplitude squared
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2
        
        # Create the k-frequency grid
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2)
        
        # Flatten the arrays to use in binning
        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        
        # Define the bins for the wavenumber
        kbins = np.arange(0.5, min(nx, ny)//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        
        # Bin the data
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                            statistic="mean",
                                            bins=kbins)
        
        # Scale the binned amplitudes
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        
        
        
        # Plotting
        
        axes[3, i].loglog(kvals, Abins, label="Simulated")

        image = pred1[i]
        
        ny, nx = image.shape
        
        # Compute the Fourier transform and get the amplitude squared
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2
        
        # Create the k-frequency grid
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2)
        
        # Flatten the arrays to use in binning
        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        
        # Define the bins for the wavenumber
        kbins = np.arange(0.5, min(nx, ny)//2+1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        
        # Bin the data
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                            statistic="mean",
                                            bins=kbins)
        
        # Scale the binned amplitudes
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        
        
        
        # Plotting
        
        axes[3, i].loglog(kvals, Abins, label="MATCHO")

        # # Adding the -5/3 slope line
        # k_ref = np.linspace(1, np.max(kbins), 100)  # Range of wavenumber for the reference line
        # energy_ref = k_ref**(-5/3)  # Energy corresponding to the -5/3 slope
        # energy_ref *= max(Abins) / max(energy_ref)  # Normalize to plot scale
        # if i != 0:
        #     axes[2, i].loglog(k_ref, energy_ref, 'k--', label='k^-5/3 Reference')


        # axes[2, i].set_xlabel('$k$')
        if i == 5:
            axes[2, i].legend()
        





    # Add row labels
    # Add row labels
    # fig.text(0.02, 0.86, 'Ground truth', va='center', ha='center', rotation='vertical', fontsize=18)
    # fig.text(0.02, 0.63, 'MATCHO', va='center', ha='center', rotation='vertical', fontsize=18)
    # fig.text(0.02, 0.38, r'Energy $P(k)$', va='center', ha='center', rotation='vertical', fontsize=18)

    # Adjust layout
    # fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.colorbar(im, cax=cbar_ax)
    # plt.show()
    plt.savefig(f"images/{epoch}.png")

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

class YourDataset_train(Dataset):
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

def preprocess_train(traj, Par):
    nsamples = traj.shape[0]
    nt = traj.shape[1]
    temp = nt - Par['lb'] - Par['lf'] + 1
    x_idx = np.arange(temp).reshape(-1,1)
    x_idx = np.tile(x_idx, (1, Par['lf'])).reshape(-1,1)

    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx+i)
    x_idx = np.concatenate(x_idx_ls, axis=1)

    t_idx = np.arange(Par['lf']).reshape(1,-1)

    t_idx = np.tile(t_idx, (temp,1)).reshape(-1,)

    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par['lb']:], window_shape=Par['lf']).reshape(-1,)

    print(f"x_idx: {x_idx.shape}")
    print(f"t_idx: {t_idx.shape}")
    print(f"y_idx: {y_idx.shape}")

    return x_idx, t_idx, y_idx

def preprocess(traj, Par):
    nsamples = traj.shape[0]
    nt = traj.shape[1]
    temp = nt - Par['lb'] - Par['LF'] + 1
    x_idx = np.arange(temp).reshape(-1,1)
    # x_idx = np.tile(x_idx, (1, Par['LF'])).reshape(-1,1)

    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx+i)
    x_idx = np.concatenate(x_idx_ls, axis=1)

    t_idx = np.arange(Par['lf']).reshape(-1,)
    # t_idx = np.tile(t_idx, (temp,1)).reshape(-1,)

    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par['lb']:], window_shape=Par['LF'])#.reshape(-1,)

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


def rollout(model, x,t,NT, Par, batch_size):
    # x - [bs, lb, nf, nx,ny]
    # t - [lf,]
    # NT - length of target time-series

    # print('NT: ', NT)

    y_pred_ls = []
    # lf = 
    # nx = 64
    # ny = 64

    bs = batch_size
    end= bs
    # for end in range(bs, x.shape[0]+1, bs):
    while True:
        start = end-bs
        out_ls = []
        
        temp_x1 = x[start:end] #[BS, lb, nf, nx,ny]
        out_ls = [temp_x1.to(device)]
        traj = torch.cat(out_ls, dim=1)

        while traj.shape[1] < NT:
            # model.eval()
            with torch.no_grad():
                temp_x = torch.repeat_interleave(temp_x1, Par['lf'], dim=0) #[BS*lf, lb, nf, nx,ny]
                temp_t = t.repeat(traj.shape[0]) #[BS*lf, ]
                with autocast():
                    out = model(temp_x.to(device), temp_t.to(device)).reshape(-1,Par['lf'], Par['nf'],Par['nx'],Par['ny']) #[BS, lf, nf, nx,ny]
                # print('out: ', out.shape)
                out_ls.append(out)
                traj = torch.cat(out_ls, dim=1)
                temp_x1 = traj[:,-Par['lb']:] #[BS, lb, nf, nx,ny]
                
        pred = torch.cat(out_ls, dim=1)[:, Par['lb']:NT] #[BS, lf, nf, nx, ny]
        # print('pred: ', pred.shape)
        y_pred_ls.append(pred)

        end = end+bs
        if end-bs > x.shape[0]+1:
            break
    
    # print("hello")
    # print(y_pred_ls)
    y_pred = torch.cat(y_pred_ls, dim=0)#.reshape(1,-1,Par['nz'],Par['ny'],Par['nx'])
    # print('y_pred: ', y_pred.shape)

    return y_pred


# Load your data into NumPy arrays (x_train, t_train, y_train, x_val, t_val, y_val, x_test, t_test, y_test)
#########################
res = 128
begin_time = time.time()
traj = np.load(f"../data/UX_nan_filtered.npy") #[nt, nx, ny]
traj = np.expand_dims(traj, axis=0) #[1, nt, nx, ny]
mask = np.load("../data/mask.npy").reshape(1,1,traj.shape[-2], traj.shape[-1])
traj = traj * mask
traj = np.expand_dims(traj, axis=2) #[1, nt, nf, nx, ny]
print(f"traj: {traj.shape}")
print(f"Data Loading Time: {time.time() - begin_time:.1f}s")


traj_train = traj[:, :800]
traj_val   = traj[:, 800:900,]
traj_test  = traj[:, 900:]

Par = {}
# Par['nt'] = 100 
Par['nx'] = traj_train.shape[-2]
Par['ny'] = traj_train.shape[-1]
Par['nf'] = 1
Par['d_emb'] = 128

Par['lb'] = 10
Par['lf'] = 2
Par['LF'] = 10
Par['channels'] = Par['nf']*Par['lb']
# Par['temp'] = Par['nt'] - Par['lb'] - Par['lf'] + 2

Par['num_epochs'] = 500 #50

time_cond = np.linspace(0, 1, Par['lf'])
if Par['lf']==1:
    time_cond = np.linspace(0, 1, Par['lf']) + 1


begin_time = time.time()
print('\nTrain Dataset')
x_idx_train, t_idx_train, y_idx_train = preprocess_train(traj_train, Par)
print('\nValidation Dataset')
x_idx_val, t_idx_val, y_idx_val  = preprocess(traj_val, Par)
print('\nTest Dataset')
x_idx_test, t_idx_test, y_idx_test  = preprocess(traj_test, Par)
print(f"Data Preprocess Time: {time.time() - begin_time:.1f}s")

# sys.exit()

t_min = np.min(time_cond)
t_max = np.max(time_cond)
if Par['lf']==1:
    t_min=0
    t_max=1


Par['inp_shift'] = np.mean(traj_train) 
Par['inp_scale'] = np.std(traj_train)
Par['out_shift'] = np.mean(traj_train)
Par['out_scale'] = np.std(traj_train)
Par['t_shift']   = t_min
Par['t_scale']   = t_max - t_min


with open('Par.pkl', 'wb') as f:
    pickle.dump(Par, f)

# sys.exit()
#########################

# Create custom datasets
mask_tensor = torch.tensor(mask, dtype=DTYPE, device=device)

Par["mask"] = mask_tensor

# Create custom datasets
traj_train_tensor = torch.tensor(traj_train, dtype=DTYPE)
traj_val_tensor = torch.tensor(traj_val, dtype=DTYPE)
traj_test_tensor = torch.tensor(traj_test, dtype=DTYPE)
time_cond_tensor = torch.tensor(time_cond, dtype=DTYPE)


train_dataset = YourDataset_train(x_idx_train, t_idx_train, y_idx_train)
val_dataset = YourDataset(x_idx_val, y_idx_val)
test_dataset = YourDataset(x_idx_test, y_idx_test)


# Define data loaders
train_batch_size = 20 #100
val_batch_size   = 20 #100
test_batch_size  = 20 #100
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

# Initialize your Unet2D model
model = Unet2D(dim=16, Par=Par, dim_mults=(1, 2, 4, 8), channels=Par['channels']).to(device).to(torch.float32)
# summary(model, input_size=((1,)+x_train.shape[1:], (1,)) )

# # Adjust the dimensions as per your model's input size
# dummy_x = x_train_tensor[0:1].to(device)
# dummy_t = t_train_tensor[0:1].to(device)
# dummy_input = (dummy_x, dummy_t)

# # Profile the model
# flops = torchprofile.profile_macs(model, dummy_input)
# print(f"FLOPs: {flops:.2e}")

# Define loss function and optimizer
criterion = CustomLoss(Par)
optimizer = optim.Adam(model.parameters(), lr=5*1e-5, weight_decay=1e-6)

# Learning rate scheduler (Cosine Annealing)
# scheduler = CosineAnnealingLR(optimizer, T_max= Par['num_epochs'] * len(train_loader) )  # Adjust T_max as needed
scheduler = combined_scheduler(optimizer, Par['num_epochs'] * len(train_loader), int(0.1 * Par['num_epochs']) * len(train_loader))


# Training loop
num_epochs = Par['num_epochs']
best_val_loss = float('inf')
best_model_id = 0

os.makedirs('models', exist_ok=True)
os.makedirs('images', exist_ok=True)

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
        for x_idx, y_idx in val_loader:
            x = traj_val_tensor[0, x_idx]        #[BS, lb, nf, nx, ny]
            t = time_cond_tensor[t_idx_val]      #[lf, ]
            y_true = traj_val_tensor[0, y_idx]   #[BS,lf, nf, nx, ny]
            y_pred = rollout(model, x,t,Par['lb']+Par['LF'], Par, val_batch_size)
            # print(f"y_true: {y_true.shape}")
            # print(f"y_pred: {y_pred.shape}")
            with autocast():
                loss   = criterion(y_pred, y_true.to(device))
            val_loss += loss.item()
        make_plot(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), epoch)

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
    for x_idx, y_idx in test_loader:
        x = traj_test_tensor[0, x_idx]        #[BS, lb, nf, nx, ny]
        t = time_cond_tensor[t_idx_test]      #[lf, ]
        y_true = traj_test_tensor[0, y_idx]   #[BS,lf, nf, nx, ny]
        y_pred = rollout(model, x,t,Par['lb']+Par['LF'], Par, val_batch_size)
        with autocast():
            loss   = criterion(y_pred, y_true.to(device))
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4e}')

