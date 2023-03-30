#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor
from models.trunks.unet import UNet
from tqdm.auto import tqdm
import dill
import seaborn as sns
import pandas as pd
import random
from scipy.stats import binom
from scipy.optimize import brentq
import pdb
import copy
from models.circles_torch_dataset import *
import os

device = torch.device('cuda:0')
alpha = 0.1
small_test = False

denoising_or_mri = 'denoising'

architecture = 'unet'
torch.cuda.empty_cache()


# Load data
sigma_noise=0.01
circles_data = noisy_circles_dataset('data/circles/n_samples_10000_n_circles_'+str(5)+'_sigma_'+str(0.5)+'_mu_'+str(0.5), 0.01, transform = ToTensor())

sigma_noise = 0.07
size = 128

class custom_dataset(Dataset):
    def __init__(self, dataset,size,noise):
        self.dataset = dataset
        self.size = size
        self.noise = noise

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset.__getitem__(idx)
        y = y.squeeze().to(device)

        y = circ_to_img(y, size = self.size)


        return y.to(device)+self.noise*torch.randn(y.shape).to(device), y.to(device)



data = custom_dataset(circles_data,size,sigma_noise)


# Set up loss and parameters for model training.
my_loss = torch.nn.MSELoss()
torch.backends.cudnn.deterministic = True
batch_size = 32
n_epochs = 40
scalings = [3,2,1]                          # Resolutions for different experiments, baseline, double, and triple
size_orig = 128                             # Size (height and width) of the original image

regs_dict = {'1':[0, 1e-3, 1e-2, 1e-1, 0.5, 0.75, 1.25],    # Different weight decays we train with for the corresponding resolutions 1, 2, and 3
            '2':[1e-3, 1e-2,1e-1,0],
            '3':[1e-2,1e-1,1e-3,0]}

for scaling in scalings:
    print(scaling)
    regs = regs_dict[str(scaling)]
    
    # initialize model
    model = UNet(n_channels_in=1, n_channels_out=1, scaling=scaling)
    size = size_orig*scaling

    # Initialize data set at the corresponding resolution and get training data (first 80% of the images).
    data = custom_dataset(circles_data,size,sigma_noise)
    training_data = torch.utils.data.Subset(data, range(int(len(data)*0.8)))

    for reg in regs:
        wd = reg/((scaling)**2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = wd)

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        model_dir = 'models/trained_models/unet/discretizations/circles/batch_size_32/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        loaded = False
        start_from = 0
        for pretrained_epoches in np.arange(n_epochs,-1,-5):
            model_path = model_dir + 'size_orig_'+str(size_orig)+'_scaling_'+str(scaling)+'_weight_decay_'+str(wd)+'_n_epochs_'+str(pretrained_epoches)+'.pt'
            try:
                model.load_state_dict(torch.load(model_path))
                start_from = pretrained_epoches
                print('Loaded model with '+str(pretrained_epoches)+ ' epochs.')
                loaded = True
                break
            except:
                pass
                
        if loaded==False or start_from<n_epochs:
            print('Have to train model from '+str(start_from)+' epochs.')
            model = model.to(device=device)
            model.train()
            for epoch in range(start_from+1,n_epochs+1):
                print('Epoch '+str(epoch)+'/'+str(n_epochs))
                for (noisy,gt) in tqdm(train_dataloader):
                    gt = gt.to(device=device)
                    noisy = noisy.to(device=device)
                    noisy = noisy.unsqueeze(1)
                    predictions = model(noisy)
                    optimizer.zero_grad()
                    l = my_loss(predictions,gt)
                    l.backward()
                    optimizer.step()
                
                if epoch%5==0:
                    model_path = model_dir + 'size_orig_'+str(size_orig)+'_scaling_'+str(scaling)+'_weight_decay_'+str(wd)+'_n_epochs_'+str(epoch)+'.pt'
                    torch.save(model.state_dict(), model_path)
            
            model_path = model_dir + 'size_orig_'+str(size_orig)+'_scaling_'+str(scaling)+'_weight_decay_'+str(wd)+'_n_epochs_'+str(n_epochs)+'.pt'
            torch.save(model.state_dict(), model_path)
    
model.eval();

