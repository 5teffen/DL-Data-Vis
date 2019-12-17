
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
import pandas as pd
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from PIL import Image

import operator


from sklearn.metrics import mean_squared_error

k = 10


input_path = './Data-Collection/Voronoi-Results/Voronoi-' + str(k) + '.csv'

file_name = "Full-file"
file_name_voronoi = "voronoi-"+str(k)
model_name = 'full_model.pth'
data_path = './data/cap2019_datavis/data'
model_path = './data/cap2019_datavis/' + model_name
output_path = './Data-Collection/'

subset = 100

simple_transform = transforms.Compose([transforms.ToTensor()])
test_data = MNIST(data_path, transform=simple_transform, download=True, train = False)
train_data = MNIST(data_path, transform=simple_transform, download=True, train = True)

# Fixing random state for reproducibility
np.random.seed(100)
my_sampler = DataSampler(np.random.permutation(len(train_data))[:subset])

test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False)

mapping = Autoenc()

mapping.load_state_dict(torch.load(model_path, map_location = 'cpu'))
mapping.eval()

input_data = None
output_data = None
reduced_data = None
labels_data = None

chosen = test_dataloader # Change if want trainining


revert_back = transforms.ToPILImage()
for i, (img, labels) in enumerate(chosen):
	batch = img.view(img.size(0), -1)
	batch = Variable(batch, requires_grad = False)
	batch_out, batch_red = mapping(batch)

	# plt.imshow(revert_back(img.squeeze()))

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


loss = nn.MSELoss()(batch_out, batch)

input_data = (input_data.cpu().detach().numpy()*255).astype(int)
output_data = (output_data.cpu().detach().numpy()*255).astype(int)
output_data[output_data < 0] = 0
reduced_data = reduced_data.cpu().detach().numpy()
labels_data = labels_data.cpu().detach().numpy()


kmeans = KMeans(n_clusters=k ,n_init=10)
kmeans.fit_transform(reduced_data)
clusters = kmeans.labels_


# -- Statistical Analysis of Clusters

clusters = np.reshape(clusters,(clusters.shape[0],1)).astype(int)
labels_data = np.reshape(labels_data,(labels_data.shape[0],1)).astype(int)

label_count = [{} for x in range(10)]
cluster_count = [{} for x in range(10)]

test_file = np.append(labels_data, clusters, axis = 1)

for f in range(test_file.shape[0]):
	label_dic = label_count[test_file[f][0]]
	cluster_no = test_file[f][1]

	if cluster_no in label_dic:
		label_dic[cluster_no] += 1
	else:
		label_dic[cluster_no] = 1

print(label_count)

for d in range(len(label_count)):
	dic = label_count[d]
	sorted_lst = max(dic.items(), key=operator.itemgetter(1))
	m = sorted_lst[0]   # Mosd
	s = sorted_lst[1]
	print("Digit " + str(d) + " -- Cluster " + str(m+1))














