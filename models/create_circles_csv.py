import torch
import pandas as pd
from sample_circles import *
import numpy as np


def create_circles_csv(path, n_samples, n_circles=5, sigma=0.5, mu=0.5):

	dataset = np.zeros([n_samples,n_circles*4])

	for i in range(n_samples):
		sample = circles(n_circles, mu = mu, sigma=sigma, device = torch.device('cpu'))
		dataset[i,:] = sample.detach().cpu().numpy().flatten()



	df = pd.DataFrame(dataset)
	df.to_csv(path+'.csv',index=False,header=None)



def create_noisy_circles_csv(path, n_samples, n_circles=5, sigma=0.5, mu=0.5, sigma_noise = 0.01):

	dataset = np.zeros([n_samples,n_circles*4])

	for i in range(n_samples):
		sample = circles(n_circles, mu = mu, sigma=sigma, device = torch.device('cpu'))
		dataset[i,:] = sample.detach().cpu().numpy().flatten()

	noisy = dataset + sigma_noise*np.random.randn(*dataset.shape)

	df = pd.DataFrame(dataset)
	df.to_csv(path+'.csv',index=False,header=None)

	df = pd.DataFrame(noisy)
	df.to_csv(path+'sigma_noise_'+str(sigma_noise)+'.csv',index=False,header=None)



n_samples = 10000

n_circles = 5
sigma = 0.5
mu = 0.5
sigma_noise = 0.01

path = '../data/circles/n_samples_'+str(n_samples)+'_n_circles_'+str(n_circles)+'_sigma_'+str(sigma)+'_mu_'+str(mu)


create_noisy_circles_csv(path, n_samples, n_circles=n_circles, sigma=sigma, mu=mu, sigma_noise = sigma_noise)





