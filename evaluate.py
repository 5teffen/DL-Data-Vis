# -- Imports -- 
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import MNIST

from my_classes import DataSampler, MyAdaptiveLR
from model_linear import Autoenc

import numpy as np
from sklearn.cluster import KMeans


data_path = './data/cap2019_datavis/data'
model_path = './data/cap2019_datavis/datavis_test1.pth'

simple_transform = transforms.Compose([transforms.ToTensor()])
data = MNIST(data_path, transform=simple_transform, download=True, train = False)
dataloader = DataLoader(data, batch_size=1000, shuffle=False)


mapping = Autoenc()
mapping.load_state_dict(torch.load(model_path))
mapping.eval()

input_data = None
output_data = None
reduced_data = None
labels_data = None

# --- Get output tensors --- 

for i, (batch, labels) in enumerate(dataloader):
	batch = batch.view(batch.size(0), -1)
	batch = Variable(batch, requires_grad = False)
	batch_out, batch_red = mapping(batch)

	if output_data is None:
		input_data = batch
		output_data = batch_out
		reduced_data = batch_red
		labels_data = labels

	else: 
		input_data = torch.cat((input_data,batch),0)
		output_data = torch.cat((output_data,batch_out),0)
		reduced_data = torch.cat((reduced_data,batch_red),0)
		labels_data = torch.cat((labels_data,labels),0)


input_data = input_data.cpu().detach().numpy() 
output_data = output_data.cpu().detach().numpy() 
reduced_data = reduced_data.cpu().detach().numpy() 
labels_data = labels_data.cpu().detach().numpy() 

kmeans = KMeans(n_clusters=3 ,n_init=10)
kmeans.fit_transform(reduced_data)
clusters = kmeans.labels_


clusters = np.reshape(clusters,(clusters.shape[0],1)).astype(int)
labels_data = np.reshape(labels_data,(labels_data.shape[0],1)).astype(int)







