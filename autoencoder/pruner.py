import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
from torchvision.datasets import MNIST

from model_linear import Autoenc_w_Pruning as Autoenc
from my_classes import DataSampler, MyAdaptiveLR
from utils import test, prune_weights, prune_rate


"""
Methodology from the paper "Learning both Weights and Connections for Efficient Neural Networks" by Han et Al. 

"""

if __name__ == "__main__":


	# -- Identify device --
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	if use_cuda:
		print("~~~ USING GPU ~~~")
	else: 
		print("~~~ USING CPU ~~~")

	# -- File paths --
	data_path = './data/cap2019_datavis/data'
	model_path = './pruning-model.pth'

	# -- Parameters --
	epochs = 10
	batch_size = 250
	learning_rate = 0.1
	momentum = 0.9
	pruning_perc = 99
	weight_decay = 5e-4


	# -- Data loaders --
	simple_transform = transforms.Compose([transforms.ToTensor()])

	data = MNIST(data_path, transform=simple_transform, train = True)
	dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

	testdata =  MNIST(data_path, transform=simple_transform, train = False)
	testdataloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)

	# -- Load trained model --
	model = Autoenc()
	model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
	model.to(device)

	print("--- Initial ---")
	test(model, testdataloader)

	# -- Removes all weights below a threshold -- 
	masks = prune_weights(model, pruning_perc)
	model.set_masks(masks)
	print("--- {}% parameters pruned ---".format(pruning_perc))
	test(model, testdataloader)

	prune_rate(model)


	# --- Retraining remainder --- 
	criterion = nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)
	adaptive_lr = MyAdaptiveLR(optimizer,factor=0.1, patience=500, stag_range= 0.0005, cooldown = 1000)

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


	test(model, testdataloader)

	# --- Save model ---

	torch.save(model.state_dict(), 'output_prune.pth')


# TODO: Prune information breakdown

# # Save and load the entire model
# torch.save(net.state_dict(), 'models/mlp_pruned.pkl')