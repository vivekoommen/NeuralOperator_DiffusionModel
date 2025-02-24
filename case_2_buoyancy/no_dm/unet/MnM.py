import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
from torch.cuda.amp import autocast

#from ml_utils import MultiHeadSelfAttention, FeedForward, FourierLayer


class Conv_Block(nn.Module):
    def __init__(self,in_c,out_c,k,num_groups=1):
        super(Conv_Block, self).__init__()
        
        self.block = nn.Sequential(
                     nn.Conv2d(in_c, out_c, k, padding='same'),
                     nn.GroupNorm(num_groups,out_c),
                     nn.GELU())

    def forward(self, x):
        return self.block(x)


class FNN_Block(nn.Module):
    def __init__(self, in_c, out_c, Nx, Ny):
        super(FNN_Block, self).__init__()

        self.Nx2 = Nx #// 2
        self.Ny2 = Ny #// 2
        self.D2 = nn.Dropout(0.5)
        # fc_output_size = 2 * in_c
        self.fc1 = nn.Linear(Nx * Ny, self.Nx2 * self.Ny2)
        self.fc2 = nn.Linear(in_c, out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), -1)  # Combined reshape operation
        x = self.relu(self.fc1(x))
        x = self.D2(x)
        x = x.transpose(1, 2)  # Transpose once
        x = self.fc2(x)
        x = x.transpose(1, 2)  # Transpose back
        x = x.view(batch_size, x.size(1), self.Nx2, self.Ny2)  # Combined reshape operation

        return x


class FNO_Block(nn.Module):
    def __init__(self, in_c, out_c, modes):
        super(FNO_Block, self).__init__()
        self.modes = modes

        self.weights_real = nn.Parameter(torch.randn(in_c, out_c, self.modes[1]-self.modes[0], self.modes[1]-self.modes[0]))
        self.weights_imag = nn.Parameter(torch.randn(in_c, out_c, self.modes[1]-self.modes[0], self.modes[1]-self.modes[0]))

        self.out_channels = out_c

    def forward(self, x):
        B, C, H, W = x.shape
        with autocast(enabled=False):
            x_ft = torch.fft.rfft2( x ).to(torch.complex64)

            out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :self.modes[1]-self.modes[0], :self.modes[1]-self.modes[0]] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, self.modes[0]:self.modes[1], self.modes[0]:self.modes[1]],
                self.weights_real + 1j * self.weights_imag
            )

            x = torch.fft.irfft2(out_ft, s=(H, W))
        return x

class MnM_Block(nn.Module):
    def __init__(self, block_type_ls, in_c, out_c, Nx, Ny, k=3, modes=6, num_groups=1):
        super(MnM_Block, self).__init__()

        self.block_ls = nn.ModuleList()

        if 'Conv' in block_type_ls:
            self.block_ls.append(Conv_Block(in_c, in_c, k, num_groups))
        if 'FNO' in block_type_ls:
            self.block_ls.append(FNO_Block(in_c, in_c, modes))
        if 'FNN' in block_type_ls:
            self.block_ls.append(FNN_Block(in_c, in_c, Nx, Ny))

        self.block_ls.append(nn.Identity())

        self.comb = nn.Conv2d(in_c * len(self.block_ls), out_c, kernel_size=1)

    def forward(self, x):
        out_ls = []
        for i in range(len(self.block_ls)):
            block = self.block_ls[i]
            out_ls.append( block(x) )
        out = torch.cat(out_ls, dim=1)
        out = self.comb(out)

        return out
    
class Enc_Block(nn.Module):
    def __init__(self, block_type_ls, in_c, out_c, lat_c, Nx, Ny, k, modes, is_down, is_lat, num_groups=1, factor=2):
        super(Enc_Block, self).__init__()

        self.is_lat = is_lat
        if is_down==True:
            self.down_sample = nn.MaxPool2d(2, stride=2)
        else:
            self.down_sample = nn.Identity()
        self.conv_block      = MnM_Block(block_type_ls, in_c, out_c, Nx, Ny, k, modes, num_groups)
        if is_lat==True:
            self.latent_block= Conv_Block(out_c,lat_c, 1, num_groups)
            
    def forward(self, x):
        down = self.down_sample(x)
        out  = self.conv_block(down)
        if self.is_lat==True:
            lat  = self.latent_block(out)
        else:
            lat = None
        return out,lat

class Dec_Block(nn.Module):
    def __init__(self, block_type_ls, in_c, out_c, lat_c, Nx, Ny, k, modes, is_lat, is_d, is_up, num_groups=1, factor=2):
        super(Dec_Block, self).__init__()
        
        temp_c = 0

        if is_lat==True:
            self.latent_block  = Conv_Block(lat_c, in_c, 1, num_groups)
            temp_c = temp_c + in_c

        if is_d==True:
            temp_c = temp_c + in_c

        self.conv_block    = MnM_Block(block_type_ls, temp_c, out_c, Nx, Ny, k, modes, num_groups)

        if is_up==True:
            self.up_sample = nn.ConvTranspose2d(out_c,out_c,kernel_size=2,stride=2)
        else:
            self.up_sample = nn.Identity()


    def forward(self,lat=None,d=None):
        # d: [B,in_c,X,Y]
        if lat is not None and d is not None:
            lat = self.latent_block(lat)
            inp = torch.cat([lat,d], axis=1)
        elif lat is not None and d is None:
            lat = self.latent_block(lat)
            inp = lat
        elif lat is None and d is not None:
            inp = d

        inp = self.conv_block(inp)            # [B,out_c,2X,2Y]
        out = self.up_sample(inp)             # [B,out_c,2X,2Y]
        return out                     


class MnM(nn.Module):
    def __init__(self,par):
        super(MnM,self).__init__()

        self.par = par
        
        n_channels = self.par['n_channels']
        k = self.par['k']
        Nx = self.par['nx']
        Ny = self.par['ny']
        block_type_ls = self.par['block_type_ls']


        # self.enc1 = MnM_Block(n_channels, 2*n_channels, k, modes1, num_heads, mlp_dim, Nx, Ny, num_groups=1) 
        # self.enc2 = MnM_Block(6*n_channels, 12*n_channels, k, modes1, num_heads, mlp_dim, 64, 64, num_groups=1)                                    
        # self.enc3 = MnM_Block(12*3*n_channels, 12*3*2*n_channels, k, modes1, num_heads, mlp_dim, 32, 32, num_groups=1)

        # self.dec3 = Dec_Block(12*3*2*n_channels,12*3*n_channels, k) 
        # self.dec2 = Dec_Block(12*n_channels,6*n_channels, k)    
        # self.dec1 = Dec_Block(2*n_channels,n_channels, k)     

        self.init_conv = Conv_Block(self.par['inp_ch'], n_channels, k=1 )

        self.enc1 = Enc_Block(block_type_ls, n_channels, n_channels,  1*n_channels, Nx//1, Ny//1, k, [24,32], is_down=False, is_lat=True)                         # [B,C,X,Y]
        self.enc2 = Enc_Block(block_type_ls, n_channels, 2*n_channels, 2*n_channels, Nx//2, Ny//2, k, [16,24], is_down=True, is_lat=True)                        # [B,2C,X/2,Y/2]
        self.enc3 = Enc_Block(block_type_ls, 2*n_channels, 4*n_channels, 4*n_channels, Nx//4, Ny//4, k, [8,16], is_down=True, is_lat=True)                     # [B,4C,X/4,Y/4]
        self.enc4 = Enc_Block(block_type_ls, 4*n_channels, 8*n_channels, 8*n_channels, Nx//8, Ny//8, k, [0,8], is_down=True, is_lat=True)                     # [B,8C,X/8,Y/8]
        self.enc5 = Enc_Block(block_type_ls, 8*n_channels, 16*n_channels, 16*n_channels, Nx//16, Ny//16, k, [0,4], is_down=True, is_lat=True)                   # [B,16C,X/16,Y/16]
        # self.enc6 = Enc_Block(16*n_channels, 32*n_channels, l_channels, k, is_down=True, is_lat=True)                 # [B,32C,X/32,Y/32]

        # self.dec6 = Dec_Block(32*n_channels, 16*n_channels, l_channels, k, is_lat=True, is_d=False, is_up=True)       # [B,16C,X/16,Y/16]
        self.dec5 = Dec_Block(block_type_ls, 16*n_channels, 8*n_channels, 16*n_channels, Nx//16, Ny//16, k, [0,4], is_lat=True, is_d=False, is_up=True)          # [B,8C,X/8,Y/8]
        self.dec4 = Dec_Block(block_type_ls, 8*n_channels, 4*n_channels, 8*n_channels, Nx//8, Ny//8, k, [0,8], is_lat=True, is_d=True, is_up=True)            # [B,4C,X/4,Y/4]
        self.dec3 = Dec_Block(block_type_ls, 4*n_channels, 2*n_channels, 4*n_channels, Nx//4, Ny//4, k, [8,16], is_lat=True, is_d=True, is_up=True)            # [B,2C,X/2,Y/2]
        self.dec2 = Dec_Block(block_type_ls, 2*n_channels, 1*n_channels, 2*n_channels, Nx//2, Ny//2, k, [16,24], is_lat=True, is_d=True, is_up=True)             # [B,C,X,Y]
        self.dec1 = Dec_Block(block_type_ls, 1*n_channels, 1*n_channels, 1*n_channels, Nx//1, Ny//1, k, [24,32], is_lat=True, is_d=True, is_up=False)            # [B,C,X,Y]
       
        self.dec0 = nn.Sequential(nn.GroupNorm(int(n_channels/4), n_channels),
                                  nn.Conv2d(n_channels,self.par['out_ch'],kernel_size=1))

    def encode(self, x):
        _, NF, LB, NX, NY = x.shape  
        x = (x-self.par['inp_shift'])/(self.par['inp_scale'])
        x = x.reshape([-1, NF*LB, NX, NY])
        
        e0 = self.init_conv(x)
        e1,l1 = self.enc1(e0)          # [B,C,X,Y]
        e2,l2 = self.enc2(e1)          # [B,2C,X/2,Y/2]
        e3,l3 = self.enc3(e2)          # [B,4C,X/4,Y/4]
        e4,l4 = self.enc4(e3)          # [B,8C,X/8,Y/18]
        e5,l5 = self.enc5(e4)          # [B,16C,X/16,Y/16]
        # e6,l6 = self.enc6(e5)          # [B,32C,X/32,Y/32]

        return [l1,l2,l3,l4,l5]
    
    def decode(self, l_ls):
        
        l1, l2, l3, l4, l5 = l_ls

        # d6 = self.dec6(l6,None)        # [B,16C,X/16,Y/16]
        d5 = self.dec5(l5,None)          # [B,8C,X/8,Y/8]
        d4 = self.dec4(l4,d5)          # [B,4C,X/4,Y/4] 
        d3 = self.dec3(l3,d4)          # [B,2C,X/2,Y/2]
        d2 = self.dec2(l2,d3)          # [B,C,X,Y]  
        d1 = self.dec1(l1,d2)          # [B,C,X,Y]

        d0 = self.dec0(d1)

        out = d0.reshape([-1, self.par["nf"], self.par["lf"], self.par["nx"], self.par["ny"]])
        out = out*self.par["out_scale"] + self.par["out_shift"]

        return out
    
    def forward(self, x):
        l_ls = self.encode(x)
        out  = self.decode(l_ls)
        
        return out
