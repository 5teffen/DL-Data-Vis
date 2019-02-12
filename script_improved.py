
# -- Imports -- 
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST
import os

import numpy as np
from rbm import RBM
from pre_training import pre_train_layer
from my_classes import DataSampler, MyAdaptiveLR
from model_linear import Autoenc,Autoenc2
from model_conv import Autoencoder as Autoenc_conv

import matplotlib
import matplotlib.pyplot as plt
# from my_dataset import myDataset


# -- Allows for GPU (cuda) utlization if available --
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True 	# Depending on whether varying input sizes


# -- Parameters --
epochs = 500			
batch_size = 100
learning_rate = 0.1
momentum = 0.9

subset = 1000


# -- Create desired transform -- 
simple_transform = transforms.Compose([
	transforms.ToTensor(),   # Converts a PIL Image/np array to a FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Range [-1,1]
])


# -- Dataset generation -- 
data = MNIST('./data', transform=simple_transform, download=True)	# Replace with custom myDataset later


# -- Sampler and Dataloader -- 
np.random.seed(0)
subset_indices = np.random.permutation(len(data))[:subset] # Create a randomized subset of indices

my_sampler = DataSampler(subset_indices)
dataloader = DataLoader(data, batch_size=batch_size, sampler = my_sampler, shuffle=False) # sampler and shuffle are not compatible



# -- Pre-Training using RBMs -- 

l1_w = pre_train_layer(dataloader,784,128,10,0.1)  # dataloader, n_in, n_out, epochs, lr





# -- Set model, loss function and optimizer --
# model = Autoenc()
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
# # adaptive_lr = MyAdaptiveLR(optimizer,factor=0.1, patience=50, stag_range= 1e-4)
# adaptive_lr = MyAdaptiveLR(optimizer,factor=0.1, patience=5, stag_range= 0.0005)


# # -- Actual Training Loop -- 

# loss_lst = []


# for epoch in range(epochs):		# Each Epoch
# 	for batch in dataloader:	# Each Iteration
# 		img, _ = batch
# 		img = img.view(img.size(0), -1)
# 		img = Variable(img, requires_grad = False)

# 		# - Forward Progragation -
# 		output,code = model(img)
# 		loss = criterion(output, img)

# 		# - Back Propagation -
# 		optimizer.zero_grad()	# Gradient needs to be reduced back to zero after each iteration
# 		loss.backward()
# 		optimizer.step()

	
# 	# - Progress Count - 
# 	loss_val = loss.data[0]
# 	loss_lst.append(loss_val)
# 	print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss_val))


# 	# - Modify LR when needed -
# 	adaptive_lr.step(loss_val,epoch)

# # print(code)

# plt.plot(list(range(1, epochs+1)), loss_lst)
# plt.xlabel("Epoch")
# plt.ylabel("MSE Loss")
# plt.show()

