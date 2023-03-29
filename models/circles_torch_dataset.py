import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor
import pandas as pd
from models.quantile_estimators import *
from models.sample_circles import *
import matplotlib
from torch.utils.data import Dataset, DataLoader

class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)

        

class circles_dataset(Dataset):

	def __init__(self, csv_file, transform=None):
		
		self.circles_frame = pd.read_csv(csv_file,header=None)
		self.transform = transform

	def __len__(self):
		return len(self.circles_frame)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		n_circles = self.circles_frame.shape[1]//4
		circles = self.circles_frame.iloc[idx,:]
		circles = np.array(circles)
		circles = circles.reshape((-1,n_circles,4))

		if self.transform:
			circles = self.transform(circles)

		return circles

	def show(self, idx):

		sample = self.__getitem__(idx)

		if sample.shape[0]==1:
			im = circ_to_img(torch.tensor(sample[0,...]),size=128)
			plt.imshow(im,cmap='gray')

		else:
			fig,axis = plt.subplots(1,sample.shape[0])

			for i in range(sample.shape[0]):
				im = circ_to_img(torch.tensor(sample[i,...]),size=128)
				axis[i].imshow(im,cmap='gray')

		plt.show()

		return


class noisy_circles_dataset(Dataset):

	def __init__(self, csv_file, sigma_noise, transform=None):
		
		self.circles_frame = pd.read_csv(csv_file+'.csv',header=None)
		self.noisy_frame = pd.read_csv(csv_file+'sigma_noise_'+str(sigma_noise)+'.csv',header=None)
		self.transform = transform

	def __len__(self):
		return len(self.circles_frame)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		n_circles = self.circles_frame.shape[1]//4
		circles = self.circles_frame.iloc[idx,:]
		circles = np.array(circles)
		circles = circles.reshape((-1,n_circles,4))

		noisy = self.noisy_frame.iloc[idx,:]
		noisy = np.array(noisy)
		noisy = noisy.reshape((-1,n_circles,4))


		if self.transform:
			circles = self.transform(circles)
			noisy = self.transform(noisy)

		sample = (noisy,circles)

		return sample

	def show(self, idx):

		sample = self.__getitem__(idx)

		if sample[0].shape[0]==1:
			feat = circ_to_img(torch.tensor(sample[0][0,...]),size=128)
			labels = circ_to_img(torch.tensor(sample[1][0,...]),size=128)
			fig,axis = plt.subplots(2)
			axis[0].imshow(feat,cmap='gray')
			axis[0].set_title('Features')
			axis[1].imshow(labels,cmap='gray')
			axis[1].set_title('Labels')


		else:
			fig,axis = plt.subplots(2,sample[0].shape[0])

			for i in range(sample[0].shape[0]):
				feat = circ_to_img(torch.tensor(sample[0][i,...]),size=128)
				labels = circ_to_img(torch.tensor(sample[1][i,...]),size=128)
				axis[0,i].imshow(feat,cmap='gray')
				axis[0,i].set_title('Features')
				axis[1,i].imshow(labels,cmap='gray')
				axis[1,i].set_title('Labels')

		plt.show()

		return






