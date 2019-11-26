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


k = 2

input_path = './Data-Collection/Voronoi-Results/Voronoi-' + str(k) + '.csv'
file_name = "Voronoi-"+str(k)
model_name = 'full_model.pth'
data_path = './data/cap2019_datavis/data'
model_path = './data/cap2019_datavis/' + model_name
output_path = './Data-Collection'


simple_transform = transforms.Compose([transforms.ToTensor()])
test_data = MNIST(data_path, transform=simple_transform, download=True, train = False)
test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False)

mapping = Autoenc()

# mapping = nn.DataParallel(mapping)
mapping.load_state_dict(torch.load(model_path, map_location = 'cpu'))
mapping.eval()


reduced_data = None

# --- Get output tensors --- 
chosen = test_dataloader 
revert_back = transforms.ToPILImage()
for i, (img, labels) in enumerate(chosen):
	batch = img.view(img.size(0), -1)
	batch = Variable(batch, requires_grad = False)
	batch_out, batch_red = mapping(batch)

	# plt.imshow(revert_back(img.squeeze()))

	if reduced_data is None:
		reduced_data = batch_red
	else: 
		reduced_data = torch.cat((reduced_data,batch_red),0)


reduced_data = reduced_data.cpu().detach().numpy()


kmeans = KMeans(n_clusters=k ,n_init=10)
kmeans.fit_transform(reduced_data)
clusters = kmeans.labels_
print(kmeans.cluster_centers_)


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
print(voronoi_output.shape)




