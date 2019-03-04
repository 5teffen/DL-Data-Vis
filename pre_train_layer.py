import numpy as np
import torch
import torchvision.datasets
from torch.autograd import Variable
import torchvision.models
import torchvision.transforms
from rbm_new import rbm


def pre_train_layer(dataloader, input_size, visible_u, hidden_u, epochs, lr, reshape=True):
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")


	cd_k = 1

	rbm_model = rbm(visible_u, hidden_u, cd_k, use_cuda=use_cuda)

	for epoch in range(epochs):
		epoch_error = 0.0

		for batch, labels in dataloader:
			# if reshape:
			batch = batch.view(len(batch), visible_u).to(device)  # flatten input data

			batch_error = rbm_model.contrastive_divergence(batch)
			
			epoch_error += batch_error

		epoch_error = epoch_error/input_size

		print('Pretraining epoch [%d], loss: %.4f' % (epoch, epoch_error))

	output_tensor = None

	# --- Outputing data --- 
	for i, (batch, labels) in enumerate(dataloader):
		batch = batch.view(len(batch), visible_u).to(device)  # flatten input data

		hidden_layer = rbm_model.sample_hidden(batch)

		if output_tensor is None:
			output_tensor = hidden_layer
		else: 
			output_tensor = torch.cat((output_tensor,hidden_layer),0)

	weight_tensor = rbm_model.weights

	return output_tensor, weight_tensor
