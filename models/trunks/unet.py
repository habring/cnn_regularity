""" Full assembly of the parts to form the complete network """
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import torch.nn.functional as F

from models.trunks.unet_parts import *
import torch.nn as nn
import pdb

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, scaling=1, bilinear=True):
        super(UNet, self).__init__()
        self.scaling = scaling
        self.n_channels_in = n_channels_in
        self.n_channels_middle = 32
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.register_buffer('lhat',None)
        self.scaling = scaling

        # path 1
        self.inc = DoubleConv(n_channels_in, 64, scaling=scaling)
        self.down1 = Down(64, 128, scaling=scaling)
        self.down2 = Down(128, 256, scaling=scaling)
        self.down3 = Down(256, 512, scaling=scaling)
        self.down4 = Down(512, 1024 // factor, scaling=scaling)

        # joined path
        self.up1 = Up(1024, 512 // factor, bilinear, scaling=scaling)
        self.up2 = Up(512, 256 // factor, bilinear, scaling=scaling)
        self.up3 = Up(256, 128 // factor, bilinear, scaling=scaling)
        self.up4 = Up(128, 64, bilinear, scaling=scaling)
        self.out = OutConv(64, self.n_channels_middle, scaling=scaling)
        self.prediction = nn.Conv2d(self.n_channels_middle, self.n_channels_out, kernel_size=scaling*3, padding=math.ceil((scaling*3-1)/2))

    def forward(self, x):
        sh = x.shape
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)

        return self.prediction(x)[:,:,:sh[2],:sh[3]].squeeze()/(self.scaling**2)