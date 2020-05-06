
# -- Imports -- 
import sys
import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../autoencoder')
from model_linear import Autoenc


def euc_dist(p1,p2):
	result = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
	return result

def closest_cluster(point, vertices):
	min_val = 100
	min_ind = -1
	for v in range(len(vertices)):
		dist = euc_dist(point, vertices[v])

		if dist < min_val:
			min_val = dist
			min_ind = v

	return min_ind


def generate_img(xyz):
	xyz = Variable(torch.FloatTensor(xyz), requires_grad = False)
	model_path = '../model/full_model.pth'
	mapping = Autoenc()
	mapping.load_state_dict(torch.load(model_path, map_location = 'cpu'))
	mapping.eval()

	generate_img = mapping.decoder(xyz)
	generate_img  = (generate_img.cpu().detach().numpy()*255).astype(int)
	generate_img[generate_img < 0] = 0

	img_size = int(math.sqrt(generate_img.shape[0]))
	data = np.reshape(generate_img, (img_size,img_size)) #reshape image
	img = toimage(data) 
	
	return data
