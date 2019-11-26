import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler


def test(model, loader):
	model.eval()

	for i, (img, labels) in enumerate(loader):
		batch = img.view(img.size(0), -1)
		batch = Variable(batch, requires_grad = False)
		batch_out, batch_red = model(batch)

	loss = nn.MSELoss()(batch_out, batch)

	print('Loss value: {:.4f}'.format(loss))
	return loss

def prune_weights(model, pruning_perc):
	all_weights = []
	for p in model.parameters():
		if len(p.data.size()) != 1:
			all_weights += list(p.cpu().data.abs().numpy().flatten())

	# -- Identifies the thershold for percentage --
	threshold = np.percentile(np.array(all_weights), pruning_perc)

	# -- Generating the mask -- 
	masks = []
	for p in model.parameters():
		if len(p.data.size()) != 1:
			pruned_inds = p.data.abs() > threshold
			masks.append(pruned_inds.float())
	return masks

def prune_rate(model, verbose=True):
	total_param = 0 # Total number of parameters
	pruned_param = 0 # Number of parameters pruned to zero

	layer_id = 0

	for parameter in model.parameters():
		layer_parameters = 1 # Number of parameters in this layer

		for dim in parameter.data.size():
			layer_parameters  *= dim

		total_param += layer_parameters 

		if len(parameter.data.size()) != 1:
			layer_id += 1
			pruned_param_layer = np.count_nonzero(parameter.cpu().data.numpy()==0)
			pruned_param += pruned_param_layer

			if verbose:
				print("Layer {} | {} layer | {:.2f}% parameters pruned" \
					.format(layer_id,'Linear',100*pruned_param_layer/layer_parameters))

	pruning_perc = 100 * pruned_param/total_param
	if verbose:
		print("Final pruning rate: {:.2f}%".format(pruning_perc))
	
	return pruning_perc



