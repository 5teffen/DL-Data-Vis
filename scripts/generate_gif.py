
import sys
sys.path.append('../')

import argparse

import math
import numpy as np
import pandas as pd

from scipy.misc import toimage, imsave

import torch
import torch.nn as nn
from torch.autograd import Variable
from model_linear import Autoenc

import imageio
from PIL import Image, ImageDraw, ImageFont

import os

# Takes in 6 arguments: x1,y1,z1 , x2,y2,z2

parser = argparse.ArgumentParser()
parser.add_argument("coorinates", metavar='N', nargs='+', type=float)
args = parser.parse_args()



def generate_img(xyz):
	xyz = Variable(torch.FloatTensor(xyz), requires_grad = False)
	model_path = '../model/full_model.pth'
	mapping = Autoenc()
	mapping.load_state_dict(torch.load(model_path, map_location = 'cpu'))
	mapping.eval()

	generate_img = mapping.decoder(xyz)
	generate_img = (generate_img.cpu().detach().numpy()*255).astype(int)
	generate_img[generate_img < 0] = 0

	img_size = int(math.sqrt(generate_img.shape[0]))
	data = np.reshape(generate_img, (img_size,img_size)) #reshape image
	img = toimage(data) 
	return img

# Inp[x1,y1,z1], [x2,y2,z2]
def transition_line(p1, p2, no_points = 50, path = "default.gif"):

	vertices = [[-1.36753164,  0.53469356, -1.69455971],
 	[-2.97084089, -2.66770672,  3.3789697 ],
 	[-2.67203856, -4.40776859, -0.25304245],
 	[-0.81943567,  5.01880814,  3.64650368],
 	[-3.39812278, -6.08068994,  6.04103366],
 	[-5.15013149,  0.08835815, -2.25413078],
 	[-1.45127244,  0.89972016,  4.0910665 ],
 	[-1.46941247, -1.83673746,  7.36687291],
 	[-3.61004374, -2.88288111, -2.93578984],
 	[-3.4752847,   1.70138008,  0.82786568]]


	out_path = "../images/transitions/" + path
	x1,y1,z1 = p1
	x2,y2,z2 = p2

	grad = [(x2-x1)/no_points, (y2-y1)/no_points, (z2-z1)/no_points]

	images = []
	for n in range(1,int(no_points)+1):
		point = [x1+grad[0]*n, y1+grad[1]*n, z1+grad[2]*n]
		image = generate_img(point)
		duration = 2

		for i in range(duration):
			images.append(image)

	# imageio.mimsave(path, images)  // Saves gif to path
	return images

if __name__ == "__main__":
	pt1 = args.coorinates[:3]
	pt2 = args.coorinates[3:]

	output_gif = transition_line(pt1, pt2)	