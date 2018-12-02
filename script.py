
# -- Imports -- 
import torch
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
# from my_dataset import myDataset


# -- Allows for GPU (cuda) utlization if available --
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True 	# Depending on whether varying input sizes


# -- Parameters --
num_epochs = 10			
batch_size = 128
learning_rate = 0.001
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
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)




# # Parameters
# params = {'batch_size': 64,
#           'shuffle': True,
#           'num_workers': 6}
# max_epochs = 100

# # Datasets
# partition = # IDs
# labels = # Labels

# # Generators
# training_set = Dataset(partition['train'], labels)
# training_generator = data.DataLoader(training_set, **params)

# validation_set = Dataset(partition['validation'], labels)
# validation_generator = data.DataLoader(validation_set, **params)

# # Loop over epochs
# for epoch in range(max_epochs):
#     # Training
#     for local_batch, local_labels in training_generator:
#         # Transfer to GPU
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#         # Model computations
#         [...]

#     # Validation
#     with torch.set_grad_enabled(False):
#         for local_batch, local_labels in validation_generator:
#             # Transfer to GPU
#             local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#             # Model computations
#             [...]