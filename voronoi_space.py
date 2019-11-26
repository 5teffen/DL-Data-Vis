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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from sklearn.metrics import mean_squared_error



# -- Parameters -- NOTE: Can replace with argparse

no_clusters = 3

point_density = 20
total = point_density**3
print("Total Volume of Points: ", total)


# -- File Paths -- 
file_name = "cluster-1"
model_name = 'epochs-10k-improved.pth'

data_path = './data/cap2019_datavis/data'
model_path = './data/cap2019_datavis/' + model_name
output_path = './data/cap2019_datavis/'


# -- Initialize Model -- 

simple_transform = transforms.Compose([transforms.ToTensor()])
test_data = MNIST(data_path, transform=simple_transform, download=True, train = False)

test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False)
mapping = Autoenc()

mapping.load_state_dict(torch.load(model_path, map_location = 'cpu'))
mapping.eval()
reduced_data = None

# --- Get Reduced Points --- 

for i, (img, labels) in enumerate(test_dataloader):
	batch = Variable(img.view(img.size(0), -1), requires_grad = False)
	batch_out, batch_red = mapping(batch)

	if reduced_data is None:
		reduced_data = batch_red
	else: 
		reduced_data = torch.cat((reduced_data,batch_red),0)

loss = nn.MSELoss()(batch_out, batch)
# print("MSE Loss:",loss.item())

reduced_data = reduced_data.cpu().detach().numpy()



# -- Identify Ranges -- 

x_range = (np.min(reduced_data[:,0]),np.max(reduced_data[:,0]))
y_range = (np.min(reduced_data[:,1]),np.max(reduced_data[:,1]))
z_range = (np.min(reduced_data[:,2]),np.max(reduced_data[:,2]))

print("X-range: ", x_range)
print("Y-range: ", y_range)
print("Z-range: ", z_range)


x_space = np.linspace(x_range[0],x_range[1],point_density)
y_space = np.linspace(y_range[0],y_range[1],point_density)
z_space = np.linspace(z_range[0],z_range[1],point_density)

# -- Create 3D space -- 

all_points = []
for x in x_space:
	for y in y_space:
		for z in z_space:
			point = np.array([x,y,z])
			all_points.append(point)

all_points = np.array(all_points)



kmeans = KMeans(n_clusters=no_clusters, n_init=10)
kmeans.fit_transform(reduced_data)
clusters = kmeans.labels_

space = kmeans.predict(all_points)



# -- Split clusters -- 
# all_clusters = []
# for c in range(no_clusters):
# 	pass

target = 2

space3d = []
for p in range(all_points.shape[0]):
	if space[p] == target:
		space3d.append(all_points[p])


fig = plt.figure(figsize=(50,50))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_zlim(z_range)
ax.set_axis_off()


colours = ['red','blue','green','orange','black','violet','cyan','gold','saddlebrown','lightpink']

for x_val, y_val, z_val in space3d:
	ax.scatter(x_val, y_val, z_val, c=colours[target], s=80, marker = "s")

# ax.legend()
plt.show()





# clusters = np.reshape(clusters,(clusters.shape[0],1)).astype(int)
# labels_data = np.reshape(labels_data,(labels_data.shape[0],1)).astype(int)

# export_file = np.append(reduced_data, labels_data, axis = 1)
# export_file = np.append(export_file, clusters, axis = 1)
# export_file_full = np.append(export_file, input_data, axis = 1)
# export_file_full = np.append(export_file_full, output_data, axis = 1)


# my_file = open(output_path + file_name + ".csv", 'ab')
# np.savetxt(my_file,export_file, delimiter=',', fmt='%f')
# my_file.close()