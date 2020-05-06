import sys
sys.path.append('../')


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

from sklearn.metrics import mean_squared_error

k = 3

data_path = '../data/cap2019_datavis/data'
model_path = '../model/full_model.pth'

simple_transform = transforms.Compose([transforms.ToTensor()])
test_data = MNIST(data_path, transform=simple_transform, download=True, train = False)
test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False)

mapping = Autoenc()

mapping.load_state_dict(torch.load(model_path, map_location = 'cpu'))
mapping.eval()

input_data = None
output_data = None
reduced_data = None
labels_data = None

# --- Get output tensors --- 


chosen = test_dataloader # Change if want trainining
# chosen = train_dataloader 

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
# print(loss.item())

input_data = (input_data.cpu().detach().numpy()*255).astype(int)
output_data = (output_data.cpu().detach().numpy()*255).astype(int)
output_data[output_data < 0] = 0
reduced_data = reduced_data.cpu().detach().numpy()
labels_data = labels_data.cpu().detach().numpy()


mse = mean_squared_error(input_data, output_data)/60000
print(mse)


kmeans = KMeans(n_clusters=k ,n_init=10)
kmeans.fit_transform(reduced_data)
clusters = kmeans.labels_
print(kmeans.cluster_centers_)


clusters = np.reshape(clusters,(clusters.shape[0],1)).astype(int)
labels_data = np.reshape(labels_data,(labels_data.shape[0],1)).astype(int)

export_file = np.append(reduced_data, clusters, axis = 1)


# --- Dealing with Voronoi --- 

all_points = pd.read_csv(input_path, header=None).values

xyz = all_points[:,:3]
labels = all_points[:,3]

clustered = kmeans.predict(xyz)

new_voronoi = []

# all_points = all_points[:,:3]
label_list = [x for x in range(k)]


for val in label_list:
	count_list = [0 for x in range(k)]
	total_count = 0
	
	for idx in range(all_points.shape[0]):
		if val == labels[idx]:
			total_count += 1
			new_label = clustered[idx]
			count_list[new_label] += 1


	to_add = count_list.index(max(count_list))

	for z in range(total_count):
		new_voronoi.append(to_add)


voronoi_output = np.append(xyz, (np.array(new_voronoi)).reshape(len(new_voronoi),1), axis = 1)


my_file = open(output_path + file_name_voronoi + ".csv", 'ab')
np.savetxt(my_file,voronoi_output, delimiter=',', fmt='%f')
my_file.close()

export_file = np.append(reduced_data, labels_data, axis = 1)
export_file = np.append(export_file, clusters, axis = 1)
export_file_full = np.append(export_file, input_data, axis = 1)
export_file_full = np.append(export_file_full, output_data, axis = 1)

my_file = open(output_path + file_name + ".csv", 'ab')
np.savetxt(my_file,export_file, delimiter=',', fmt='%f')
my_file.close()


my_file = open(output_path + file_name + "_full.csv", 'ab')
np.savetxt(my_file,export_file_full, delimiter=',', fmt='%f')
my_file.close()


plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')  