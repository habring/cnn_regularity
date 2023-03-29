import numpy as np
import torch
import matplotlib.pyplot as plt


torch.manual_seed(0)
def circ_to_img(circs, size = 56):

	if circs.get_device()==-1:
		device = torch.device('cpu')
	else:
		device = torch.device('cuda:0')

	im = torch.zeros([size,size], device = device)
	
	for i in range(circs.shape[0]):
		X = torch.arange(0,size,1,device=device)
		Y = torch.arange(0,size,1,device=device)
		YY,XX = torch.meshgrid(Y,X)
		dist_from_center = 1.0*(torch.sqrt((XX - torch.floor(circs[i,0]*size))**2 + (YY-torch.floor(circs[i,1]*size))**2)<=(torch.floor(circs[i,2]*size)))
		im += dist_from_center*circs[i,3]

	return im


def circles(n_circles, mu = 0.5, sigma=0.5, show=False, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

	circs = torch.zeros([n_circles,4])

	circs = mu + sigma*torch.randn([n_circles,4])
	#circs[:,2] = 0.5 + sigma*torch.randn([n_circles])
	#circs[:,3] = 0.5 + sigma*torch.randn([n_circles])


	if show:
		im = circ_to_img(circs)
		plt.imshow(im,cmap='gray')
		plt.show()

	return circs


