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
from sklearn.cluster import KMeans

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
if use_cuda:
	print("~~~ USING GPU ~~~")
device = torch.device("cuda:0" if use_cuda else "cpu")

# torch.backends.cudnn.benchmark = True 	# Depending on whether varying input sizes
# if torch.cuda.is_available():
# 	torch.set_gpu_as_default_device()

# -- Parameters --
epochs = 10
batch_size = 250
learning_rate = 0.1
momentum = 0.9

subset = 60000  # How many samples of the whole dataset worked on


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


# -- Set model, loss function and optimizer --
dataloader = DataLoader(data, batch_size=batch_size, sampler = my_sampler, shuffle=False) # sampler and shuffle are not compatible

model = Autoenc().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
adaptive_lr = MyAdaptiveLR(optimizer,factor=0.1, patience=100, stag_range= 0.0005)


# -- Pre-Training using RBMs -- 

# new: 784 -> 194 -> 48 -> 12 -> 3

# epochs_pretrain = 5

# rbm0 = DataLoader(data, batch_size=subset, sampler = my_sampler, shuffle=False)
# h1, l1_w, l1_bias = pre_train_layer(rbm0,784,194,epochs_pretrain,0.01)  # dataloader, n_in, n_out, epochs, lr

# rbm1 = TensorDataset(h1,h1) # input data, target
# h2, l2_w, l2_bias = pre_train_layer(rbm1,194,48,epochs_pretrain,0.1) 

# rbm2 = TensorDataset(h2,h2)
# h3, l3_w, l3_bias = pre_train_layer(rbm2,48,12,epochs_pretrain,0.1) 

# rbm3 = TensorDataset(h3,h3)
# h4, l4_w, l4_bias = pre_train_layer(rbm3,12,3,epochs_pretrain,0.1) 


# -- Manually Modifying Weights -- 
# state_dict = model.state_dict()

# print(state_dict['decoder.0.weight'].size())
# print(state_dict['decoder.2.weight'].size())
# print(state_dict['decoder.4.weight'].size())
# print(state_dict['decoder.6.weight'].size())

# print(l4_w.t().size())
# print(l3_w.t().size())
# print(l2_w.t().size())
# print(l1_w.t().size())

# state_dict['encoder.0.weight'] = l1_w  # layer 1 weight
# state_dict['encoder.2.weight'] = l2_w
# state_dict['encoder.4.weight'] = l3_w
# state_dict['encoder.6.weight'] = l4_w

# state_dict['decoder.0.weight'] = l4_w.t()
# state_dict['decoder.2.weight'] = l3_w.t()
# state_dict['decoder.4.weight'] = l2_w.t()
# state_dict['decoder.6.weight'] = l1_w.t()

# model.load_state_dict(state_dict)


# -- Actual Training Loop -- 
loss_lst = []

# print(data[1][0].size())
# print(data[1][0].size())



for epoch in range(epochs):		# Each Epoch
	for batch in dataloader:	# Each Iteration
		img_data, label = batch
		img_data = img_data.view(img_data.size(0), -1)
		img_data = Variable(img_data, requires_grad = False).to(device)

		# - Forward Progragation -
		output,code = model(img_data)
		loss = criterion(output, img_data)

		# - Back Propagation -
		optimizer.zero_grad()	# Gradient needs to be reduced back to zero after each iteration
		loss.backward()
		optimizer.step()

	
	# - Progress Count - 
	loss_val = loss.item()
	loss_lst.append(loss_val)
	print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss_val))


	# - Modify LR when needed -
	adaptive_lr.step(loss_val,epoch)


# --- Get final code ---   REQUIRES FIX
all_loader = DataLoader(data, batch_size=subset)
for batch in all_loader:
	all_data, all_labels = batch
all_data = all_data.view(all_data.size(0), -1)
all_data = Variable(all_data, requires_grad = False)

all_output, all_code = model(all_data)

# --- Save model ---
torch.save(model.state_dict(), './datavis_test1.pth')

"""
Batch problem
Cuda implementation
Tests
K-means clustering


use .cuda() on any input batches/tensors

use .cuda() on your network module, which will hold your network, like:
"""

# # print(code)

my_file = open('testing.csv', 'ab')

code_np = all_code.detach().numpy()
labels_np = all_labels.detach().numpy()

kmeans = KMeans(n_clusters=3 ,n_init=10)
kmeans.fit(code_np)
clusters = kmeans.labels_

# --- Reshaping to export --- 

clusters = np.reshape(clusters,(clusters.shape[0],1))
labels_np = np.reshape(labels_np,(labels_np.shape[0],1))

# print("Shapes 1:")
# print(code_np.shape)
# print(labels_np.shape)
# print(clusters.shape)

labeled_data = np.append(code_np, labels_np, axis = 1)
clustered_data = np.append(code_np, clusters, axis = 1)
all_data = np.append(labeled_data, clusters, axis = 1)

# print("Shapes 2:")
# print(labeled_data.shape)
# print(clustered_data.shape)
# print(all_data.shape)

my_file = open('data/cap2019_datavis/w-labels.csv', 'ab')
np.savetxt(my_file,labeled_data, delimiter=',')
my_file.close()

my_file = open('data/cap2019_datavis/w-clusters.csv', 'ab')
np.savetxt(my_file,clustered_data, delimiter=',')
my_file.close()

my_file = open('data/cap2019_datavis/w-all.csv', 'ab')
np.savetxt(my_file,all_data, delimiter=',')
my_file.close()


plt.plot(list(range(1, epochs+1)), loss_lst)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig('data/cap2019_datavis/testing.png')
#plt.show()

