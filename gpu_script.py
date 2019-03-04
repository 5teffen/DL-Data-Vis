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

from pre_train_layer import pre_train_layer
from my_classes import DataSampler, MyAdaptiveLR
from model_linear import Autoenc,Autoenc2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from my_dataset import myDataset

# -- Allows for GPU (cuda) utlization if available --
use_cuda = torch.cuda.is_available()
if use_cuda:
	print("~~~ USING GPU ~~~")
else: 
	print("~~~ USING CPU ~~~")

device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True 	# Depending on whether varying input sizes
# if torch.cuda.is_available():
# 	torch.set_gpu_as_default_device()



# -- Parameters --
epochs = 500
batch_size = 250
learning_rate = 0.1
momentum = 0.9

subset = 60000  # How many samples of the whole dataset worked on

data_path = './data/cap2019_datavis/data'
output_path = './data/cap2019_datavis/'
output_file = 'test_norm1'


# -- Create desired transform -- 
simple_transform = transforms.Compose([
	transforms.ToTensor(),   # Converts a PIL Image/np array to a FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Range [-1,1]
])


# -- Dataset generation -- 
data = MNIST(data_path, transform=simple_transform, download=True, train = True)	# Replace with custom myDataset later


# -- Define Sampler and take Subset -- 
np.random.seed(0)
subset_indices = np.random.permutation(len(data))[:subset] # Create a randomized subset of indices
my_sampler = DataSampler(subset_indices)


# -- Set model, loss function and optimizer --
dataloader = DataLoader(data, batch_size=batch_size, sampler = my_sampler, shuffle=False) # sampler and shuffle are not compatible
# dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
model = Autoenc().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
adaptive_lr = MyAdaptiveLR(optimizer,factor=0.1, patience=200, stag_range= 0.0005, cooldown = 100)


# -- Pre-Training using RBMs -- 

# new: 784 -> 194 -> 48 -> 12 -> 3
# batch_pretrain = 100
# epochs_pretrain = 100

# rbm0 = DataLoader(data, batch_size=batch_pretrain, shuffle=False)
# hidden1, weights1 = pre_train_layer(rbm0,subset,784,194,epochs_pretrain,0.01)

# rbm1 = DataLoader(TensorDataset(hidden1,hidden1), batch_size=batch_pretrain, shuffle=False)
# hidden2, weights2 = pre_train_layer(rbm1,subset,194,48,epochs_pretrain,0.01,False)

# rbm2 = DataLoader(TensorDataset(hidden2,hidden2), batch_size=batch_pretrain, shuffle=False)
# hidden3, weights3 = pre_train_layer(rbm2,subset,48,12,epochs_pretrain,0.01,False)

# rbm3 = DataLoader(TensorDataset(hidden3,hidden3), batch_size=batch_pretrain, shuffle=False)
# hidden4, weights4 = pre_train_layer(rbm3,subset,12,3,epochs_pretrain,0.01,False)


# # -- Manually Modifying Weights -- 
# state_dict = model.state_dict()


# state_dict['encoder.0.weight'] = weights1.t() # layer 1 weight
# state_dict['encoder.2.weight'] = weights2.t()
# state_dict['encoder.4.weight'] = weights3.t()
# state_dict['encoder.6.weight'] = weights4.t()

# state_dict['decoder.0.weight'] = weights4
# state_dict['decoder.2.weight'] = weights3
# state_dict['decoder.4.weight'] = weights2
# state_dict['decoder.6.weight'] = weights1

# model.load_state_dict(state_dict)
# model.eval()


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

# --- Get final code ---   REQUIRES FIX

all_loader = DataLoader(data, batch_size=subset)
for batch in all_loader:
	all_data, all_labels = batch
all_data = all_data.view(all_data.size(0), -1)
all_data = Variable(all_data, requires_grad = False).to(device)
final_out, final_code = model(all_data)


code_np = final_code.cpu().detach().numpy()
labels_np = all_labels.cpu().detach().numpy()

kmeans = KMeans(n_clusters=3 ,n_init=10)
kmeans.fit(code_np)
clusters = kmeans.labels_


# --- Reshaping to export --- 

clusters = np.reshape(clusters,(clusters.shape[0],1)).astype(int)
labels_np = np.reshape(labels_np,(labels_np.shape[0],1)).astype(int)

labeled_data = np.append(code_np, labels_np, axis = 1)
clustered_data = np.append(code_np, clusters, axis = 1)
all_data = np.append(labeled_data, clusters, axis = 1)


# --- Export ---

my_file = open(output_path + output_file +'.csv', 'ab')
np.savetxt(my_file,all_data, delimiter=',')
my_file.close()


plt.plot(list(range(1, epochs+1)), loss_lst)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig(output_path + output_file + '.png')
#plt.show()

