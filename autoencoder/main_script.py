# -- Imports -- 
import sys
sys.path.append('../')
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

from pre_train_layer import pre_train_layer
from my_classes import DataSampler, MyAdaptiveLR
from model_linear import Autoenc,Autoenc2
from model_linear import Autoenc_w_Pruning as Auto

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -- Allows for GPU (cuda) utlization if available --
use_cuda = torch.cuda.is_available()
if use_cuda:
	print("~~~ USING GPU ~~~")
else: 
	print("~~~ USING CPU ~~~")

device = torch.device("cuda:0" if use_cuda else "cpu")

# -- Parameters --
epochs = 100000
batch_size = 200
learning_rate = 0.1
momentum = 0.9

subset = 60000  # How many samples of the whole dataset worked on

# data_path = '../data/cap2019_datavis/data'
# output_path = '../data/cap2019_datavis/'

data_path = '../data/'
output_path = '../data/output/'


output_file = 'mega-model'


# -- Create desired transform -- 
simple_transform = transforms.Compose([
	transforms.ToTensor(),   # Converts a PIL Image/np array to a FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Range [-1,1]
])


# -- Dataset generation -- 
data = MNIST(data_path, transform=simple_transform, download=True, train = True)	# Replace with custom myDataset later


# -- Define Sampler and take Subset -- 
np.random.seed(0) # Fixing random state for reproducibility
subset_indices = np.random.permutation(len(data))[:subset] # Create a randomized subset of indices
my_sampler = DataSampler(subset_indices)


# -- Set model, loss function and optimizer --
# dataloader = DataLoader(data, batch_size=batch_size, sampler = my_sampler, shuffle=False) # sampler and shuffle are not compatible
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
model = Auto().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
adaptive_lr = MyAdaptiveLR(optimizer,factor=0.1, patience=5000, stag_range= 0.0005, cooldown = 5000)


# -- Actual Training Loop -- 
loss_lst = []

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
	print('Training epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss_val))


	# - Modify LR when needed -
	adaptive_lr.step(loss_val,epoch)


# --- Save model ---

torch.save(model.state_dict(), output_path + output_file +'.pth')

plt.plot(list(range(1, epochs+1)), loss_lst)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig(output_path + output_file + '.png')

#plt.show()

