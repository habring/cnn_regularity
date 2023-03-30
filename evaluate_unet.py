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
import matplotlib.gridspec as gridspec
from pathlib import Path


device = torch.device('cpu')
torch.cuda.empty_cache()

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


# load the data
circles_data = noisy_circles_dataset('data/circles/n_samples_10000_n_circles_'+str(5)+'_sigma_'+str(0.5)+'_mu_'+str(0.5), 0.01, transform = ToTensor())
sigma_noise = 0.07
size = 64

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


# Size (height and width) of the test images
size_orig = 128

# Pick the samples and parameters shown in the paper.
sample_idx = torch.tensor([597, 1675, 982,  545, 1923])
n_epochs = 40
regs = [0, 1e-3,1e-2, 1e-1] # weight decay values
scalings = [1,2,3]          # different resolutions, baseline, double, and triple



# directory to save images in

output_dir = 'data/results/unet/'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# define subplots
widths = [1]*6
heights = [1,0.5]*5
fig_different_weightdecays = plt.figure(tight_layout=True,figsize=(20,20))
gs_different_weightdecays = gridspec.GridSpec(10,6,hspace = 0, wspace = 0, width_ratios=widths,height_ratios=heights)

widths = [1]*5
heights = [1,0.5]*5
fig_different_resolutions = plt.figure(tight_layout=True,figsize=(20,20))
gs_different_resolutions = gridspec.GridSpec(10,5,hspace = 0, wspace = 0, width_ratios=widths,height_ratios=heights)

# plotting parameters
col=(31/255,76/255,132/255)     # color for the 1D line plots
height = [ 10, 20, 65, 122,55]  # x-axis coordinates for the cuts through the images
ms = 5                          # marker size for the 1D plots

with torch.no_grad():
    for i in range(len(sample_idx)):

        # initialize torch data set
        data = custom_dataset(circles_data,size_orig,sigma_noise)
        testing_data = torch.utils.data.Subset(data, range(int(len(data)*0.8),len(data)))
        noisy,img = testing_data[sample_idx[i].item()]
        img=img.to(device)
        noisy=noisy.to(device)

        img_min = img.min().item()
        img_max = img.max().item()

        # Plot noisy image
        im_noisy = copy.deepcopy(noisy.squeeze().detach().cpu().numpy())
        im_noisy = (np.clip(im_noisy,img_min,img_max)-img_min)/(img_max-img_min)
        im_noisy = np.concatenate([im_noisy[...,np.newaxis]]*3,axis=2)
        im_noisy[height[i],:,0] = 1
        im_noisy[height[i],:,1] = 0
        im_noisy[height[i],:,2] = 0
        ax = fig_different_weightdecays.add_subplot(gs_different_weightdecays[2*i,0])
        ax.set_xticks([])
        ax.set_yticks([])
        if i==0:
            ax.set_title('Noisy')
        ax.imshow(im_noisy)

        ax = fig_different_resolutions.add_subplot(gs_different_resolutions[2*i,0])
        ax.set_xticks([])
        ax.set_yticks([])
        if i==0:
            ax.set_title('Noisy')
        ax.imshow(im_noisy)
        plt.imsave(output_dir+'image_'+str(i)+'_noisy.png',im_noisy)


        # Plot cut through noisy image
        im = noisy.squeeze().detach().cpu().numpy()
        cut_noisy = copy.deepcopy(im[height[i],:])
        y_min = cut_noisy.min()-0.5
        y_max = cut_noisy.max()+0.5

        g = plt.figure()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.ylim(y_min,y_max)
        plt.plot(cut_noisy,'.',markersize = ms,color=col)
        plt.savefig(output_dir+'image_'+str(i)+'_noisy_cut.png')
        plt.close(g)

        ax = fig_different_weightdecays.add_subplot(gs_different_weightdecays[2*i+1,0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(.5)
        ax.plot(cut_noisy,'.',markersize = ms,color=col)
        ax.set_ylim(y_min,y_max)

        ax = fig_different_resolutions.add_subplot(gs_different_resolutions[2*i+1,0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(.5)
        ax.plot(cut_noisy,'.',markersize = ms,color=col)
        ax.set_ylim(y_min,y_max)

        # Plot ground truth
        im_gt = copy.deepcopy(img.squeeze().detach().cpu().numpy())
        im_gt = (im_gt-img_min)/(img_max-img_min)
        im_gt = np.concatenate([im_gt[...,np.newaxis]]*3,axis=2)
        im_gt[height[i],:,0] = 1
        im_gt[height[i],:,1] = 0
        im_gt[height[i],:,2] = 0
        plt.imsave(output_dir+'image_'+str(i)+'_ground.png',im_gt)
        ax = fig_different_weightdecays.add_subplot(gs_different_weightdecays[2*i,1])
        ax.set_xticks([])
        ax.set_yticks([])
        if i==0:
            ax.set_title('Ground truth')
        ax.imshow(im_gt)

        ax = fig_different_resolutions.add_subplot(gs_different_resolutions[2*i,1])
        ax.set_xticks([])
        ax.set_yticks([])
        if i==0:
            ax.set_title('Ground truth')
        ax.imshow(im_gt)


        # Plot cut through ground truth
        cut_gt = copy.deepcopy(img.squeeze().detach().cpu().numpy()[height[i],:])

        g = plt.figure()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.ylim(y_min,y_max)
        plt.plot(cut_gt,'.',markersize = ms,color=col)
        plt.savefig(output_dir+'image_'+str(i)+'_ground_cut.png')
        plt.close(g)
        ax = fig_different_weightdecays.add_subplot(gs_different_weightdecays[2*i+1,1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(.5)
        ax.plot(cut_gt,'.',markersize = ms,color=col)
        ax.set_ylim(y_min,y_max)

        ax = fig_different_resolutions.add_subplot(gs_different_resolutions[2*i+1,1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(.5)
        ax.plot(cut_gt,'.',markersize = ms,color=col)
        ax.set_ylim(y_min,y_max)


        for i_reg,reg in enumerate(regs):
            for i_scaling,scaling in enumerate(scalings):

                if scaling==1 or reg == 0.001:

                    # Initialize torch data set for corresponding resolution
                    data = custom_dataset(circles_data,size_orig*scaling,sigma_noise)
                    testing_data = torch.utils.data.Subset(data, range(int(len(data)*0.8),len(data)))
                    noisy,img = testing_data[sample_idx[i].item()]

                    # Get the pre-trained U-net for the corresponding resolution and compute output
                    model = UNet(n_channels_in=1, n_channels_out=1, scaling=scaling)
                    model_dir = 'models/trained_models/unet/discretizations/circles/batch_size_32/'
                    model.eval()
                    model.to(device)
                    wd = (reg/(scaling)**2)
                    model_name = 'size_orig_'+str(size_orig)+'_scaling_'+str(scaling)+'_weight_decay_'+str(wd)+'_n_epochs_'+str(n_epochs)+'.pt'
                    model_path = model_dir + model_name
                    try:
                        model.load_state_dict(torch.load(model_path,map_location=device))
                    except:
                        print('There are no pretrained models available. We download the weights from Zenodo...')
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        os.system("wget -O " + model_path + " https://zenodo.org/record/7784039/files/" + model_name)
                        model.load_state_dict(torch.load(model_path,map_location=device))
                    recon = model(noisy.unsqueeze(0).unsqueeze(0)).detach().cpu().numpy()

                    # Plot output image
                    im_recon = copy.deepcopy(recon)
                    im_recon = (np.clip(im_recon,img_min,img_max)-img_min)/(img_max-img_min)
                    im_recon = np.concatenate([im_recon[...,np.newaxis]]*3,axis=2)
                    im_recon[height[i]*scaling,:,0] = 1
                    im_recon[height[i]*scaling,:,1] = 0
                    im_recon[height[i]*scaling,:,2] = 0
                    if scaling==1:
                        ax = fig_different_weightdecays.add_subplot(gs_different_weightdecays[2*i,2+i_reg])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if i==0:
                            ax.set_title('Weight decay: '+str(reg))
                        ax.imshow(im_recon)

                    if reg == 0.001:
                        ax = fig_different_resolutions.add_subplot(gs_different_resolutions[2*i,2+i_scaling])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if i==0:
                            ax.set_title('Resolution: '+str(scaling))
                        ax.imshow(im_recon)
                    plt.imsave((output_dir+'size_orig_'+str(size_orig)+'_scaling_'+str(scaling)+'_weight_decay_'+str(wd)+'_n_epochs_'+str(n_epochs)+'_image_'+str(i)).replace('.',',')+'.png',im_recon)


                    # Plot cut through output image
                    cut_recon = copy.deepcopy(recon[height[i]*scaling,:])
                    g = plt.figure()
                    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
                    plt.margins(0,0)
                    plt.ylim(y_min,y_max)
                    plt.plot(cut_recon,'.',markersize = ms,color=col)
                    plt.savefig((output_dir+'size_orig_'+str(size_orig)+'_scaling_'+str(scaling)+'_weight_decay_'+str(wd)+'_n_epochs_'+str(n_epochs)+'_image_'+str(i)).replace('.',',')+'_cut.png')
                    plt.close(g)
                    if scaling==1:
                        ax = fig_different_weightdecays.add_subplot(gs_different_weightdecays[2*i+1,2+i_reg])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_box_aspect(.5)
                        ax.plot(cut_recon,'.',markersize = ms,color=col)
                        ax.set_ylim(y_min,y_max)

                    if reg == 0.001:
                        ax = fig_different_resolutions.add_subplot(gs_different_resolutions[2*i+1,2+i_scaling])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_box_aspect(.5)
                        ax.plot(cut_recon,'.',markersize = ms,color=col)
                        ax.set_ylim(y_min,y_max)

fig_different_weightdecays.savefig(output_dir+'comparison_weight_decays.png')
fig_different_resolutions.savefig(output_dir+'comparison_resolutions.png')