
# -- Imports -- 
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
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
epochs = 200			
batch_size = 100
learning_rate = 0.1
momentum = 0.9

subset = 1000  # How many samples of the whole dataset worked on


# -- Create desired transform -- 
simple_transform = transforms.Compose([
	transforms.ToTensor(),   # Converts a PIL Image/np array to a FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Range [-1,1]
])


# -- Dataset generation -- 
data = MNIST('./data', transform=simple_transform, download=True)	# Replace with custom myDataset later


# -- Define Sampler and take Subset -- 
np.random.seed(0)
subset_indices = np.random.permutation(len(data))[:subset] # Create a randomized subset of indices
my_sampler = DataSampler(subset_indices)


# -- Pre-Training using RBMs -- 
epochs_pretrain = 3

rbm0 = DataLoader(data, batch_size=subset, sampler = my_sampler, shuffle=False)
h1, l1_w, l1_bias = pre_train_layer(rbm0,784,128,epochs_pretrain,0.01)  # dataloader, n_in, n_out, epochs, lr

rbm1 = TensorDataset(h1,h1) # input data, target
h2, l2_w, l2_bias = pre_train_layer(rbm1,128,64,epochs_pretrain,0.1) 

rbm2 = TensorDataset(h2,h2)
h3, l3_w, l3_bias = pre_train_layer(rbm2,64,12,epochs_pretrain,0.1) 

rbm3 = TensorDataset(h3,h3)
h4, l4_w, l4_bias = pre_train_layer(rbm3,12,3,epochs_pretrain,0.1) 


# -- Set model, loss function and optimizer --
dataloader = DataLoader(data, batch_size=batch_size, sampler = my_sampler, shuffle=False) # sampler and shuffle are not compatible

model = Autoenc()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
# # adaptive_lr = MyAdaptiveLR(optimizer,factor=0.1, patience=50, stag_range= 1e-4)
adaptive_lr = MyAdaptiveLR(optimizer,factor=0.1, patience=5, stag_range= 0.0005)


# -- Actual Training Loop -- 
loss_lst = []

# -- Manually Modifying Weights -- 
state_dict = model.state_dict()

print(state_dict['decoder.0.weight'].size())
# print(state_dict['decoder.2.weight'].size())
# print(state_dict['decoder.4.weight'].size())
# print(state_dict['decoder.6.weight'].size())

# print(l4_w.t().size())
# print(l3_w.t().size())
# print(l2_w.t().size())
# print(l1_w.t().size())

state_dict['encoder.0.weight'] = l1_w  # layer 1 weight
state_dict['encoder.2.weight'] = l2_w
state_dict['encoder.4.weight'] = l3_w
state_dict['encoder.6.weight'] = l4_w

state_dict['decoder.0.weight'] = l4_w.t()
state_dict['decoder.2.weight'] = l3_w.t()
state_dict['decoder.4.weight'] = l2_w.t()
state_dict['decoder.6.weight'] = l1_w.t()

model.load_state_dict(state_dict)


for epoch in range(epochs):		# Each Epoch
	for batch in dataloader:	# Each Iteration
		img, _ = batch
		img = img.view(img.size(0), -1)
		img = Variable(img, requires_grad = False)

		# - Forward Progragation -
		output,code = model(img)
		loss = criterion(output, img)

		# - Back Propagation -
		optimizer.zero_grad()	# Gradient needs to be reduced back to zero after each iteration
		loss.backward()
		optimizer.step()

	
	# - Progress Count - 
	loss_val = loss.data[0]
	loss_lst.append(loss_val)
	print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss_val))


	# - Modify LR when needed -
	adaptive_lr.step(loss_val,epoch)


"""
Batch problem
Cuda implementation
Tests
K-means clustering 
"""

# # print(code)

# plt.plot(list(range(1, epochs+1)), loss_lst)
# plt.xlabel("Epoch")
# plt.ylabel("MSE Loss")
# plt.show()

