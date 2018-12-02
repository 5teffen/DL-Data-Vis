
# -- Imports -- 
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import os

import numpy as np
from my_classes import DataSampler
from model_linear import autoenc

import matplotlib
import matplotlib.pyplot as plt
# from my_dataset import myDataset


# -- Allows for GPU (cuda) utlization if available --
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True 	# Depending on whether varying input sizes


# -- Parameters --
epochs = 1000			
batch_size = 150
learning_rate = 0.005
momentum = 0.9


# -- Create desired transform -- 
simple_transform = transforms.Compose([
	transforms.ToTensor(),   # Converts a PIL Image/np array to a FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Range [-1,1]
])


# -- Dataset generation -- 
data = MNIST('./data', transform=simple_transform, download=True)	# Replace with custom myDataset later


# -- Sampler and Dataloader -- 
np.random.seed(0)
subset_indices = np.random.permutation(len(data))[:5000] # Create a randomized subset of indices

my_sampler = DataSampler(subset_indices)
dataloader = DataLoader(data, batch_size=batch_size, sampler = my_sampler, shuffle=False) # sampler and shuffle are not compatible


# -- Set model, loss function and optimizer --
model = autoenc()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)


# -- Training loop -- 

loss_lst = []


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
	loss_lst.append(loss.data[0])
	print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss.data[0]))



plt.plot(list(range(1, epochs+1)), loss_lst)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()


