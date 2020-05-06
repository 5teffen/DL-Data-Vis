"""
Scripts for performing the evaluation of the dimensionality reduction

Functionality:
- Averaging cluster images
- Single feature column visualisation
- Highlight difference in images

"""

import sys
sys.path.append('../')


import pyvoro
import math
import numpy as np
import pandas as pd

from scipy.misc import toimage, imsave, imshow

import torch
import torch.nn as nn
from torch.autograd import Variable
from model_linear import Autoenc

import imageio
from PIL import Image, ImageDraw, ImageFont

import os

from exploration_utils import *



def cluster_avg(point_lst):
	shape = generate_img(point_lst[0]).shape
	no_pts = len(point_lst)
	sum_img = np.zeros(shape)

	for pt in point_lst:
		one_img = generate_img(pt)
		sum_img = np.add(sum_img, one_img)

	avg_img = sum_img/no_pts

	return avg_img


def populate_cuboid(x_range, y_range, z_range, no_points):

   x = np.random.uniform(x_range[0],x_range[1],(no_points,))
   y = np.random.uniform(y_range[0],y_range[1],(no_points,))
   z = np.random.uniform(z_range[0],z_range[1],(no_points,))

   return (x,y,z)


def points_in_cluster(x_points,y_points,z_points, clusters, target):
	pt_lst = []
	for i in range(len(x_points)):
		a_point = [x_points[i], y_points[i], z_points[i]]
		min_cluster = closest_cluster(a_point, clusters)
		if (min_cluster == target):
			pt_lst.append(list(a_point))

	return pt_lst



# ==== Main Loop ====
if __name__ == '__main__':


	# --- Pre-recorded results --- 

	c10 = [[-1.36753164,  0.53469356, -1.69455971],
	[-2.97084089, -2.66770672,  3.3789697 ],
	[-2.67203856, -4.40776859, -0.25304245],
	[-0.81943567,  5.01880814,  3.64650368],
	[-3.39812278, -6.08068994,  6.04103366],
	[-5.15013149,  0.08835815, -2.25413078],
	[-1.45127244,  0.89972016,  4.0910665 ],
	[-1.46941247, -1.83673746,  7.36687291],
	[-3.61004374, -2.88288111, -2.93578984],
	[-3.4752847,   1.70138008,  0.82786568]]



 # 	c10 = [[-3.31521574, -5.89149675,  5.81095282],
	# [-1.41730071,  0.84941566, 4.01813861],
	# [-2.6455994,  -4.33521574, -0.24320068],
	# [-3.36601638,  1.62761706,  0.7966109 ],
	# [-0.82151942,  4.85083063,  3.54361298],
	# [-2.98348292, -2.52781751,  3.32872026],
	# [-5.06600251,  0.02730934, -2.25792916],
	# [-1.52776481, -1.83322744,  7.19034007],
	# [-1.30862023,  0.5653143,  -1.67967383],
	# [-3.35278849, -2.76668423, -2.74202434]]


	c10 = np.sort(c10)


	range_x = (-10.494617, 2.9902372)
	range_y = (-12.401706, 12.255847)
	range_z = (-7.081162, 12.459479)

	test_xyz = [-2.65846443, -4.3750658, -0.30079758]


	# --- Load datapoints ---
	code_data = pd.read_csv("../data/working-data/reduced-data.csv", index_col = False, header = None).values
	x_data = code_data[:,0]
	y_data = code_data[:,1]
	z_data = code_data[:,2]

	no_clstrs = len(c10)

	# --- Generate arbitrary point cloud --- 
	pt_density = 1000
	x_pts,y_pts,z_pts = populate_cuboid(range_x,range_y,range_z,pt_density)



	# # === (1) Averaging of points in clusters ===
	# for c in range(no_clstrs):

	# 	# -- Identify points that fall within a single cluster --
	# 	pt_lst = points_in_cluster(x_data,y_data,z_data, c10, c)
		
	# 	# -- Find average of point list --
	# 	img_data = cluster_avg(pt_lst)
	# 	img = toimage(img_data) 
	# 	# imsave('images/clusters/cp_' + str(c) +'.jpg', img)



	# # === (2) Averaging of all 3D space ===
	# for c in range(no_clstrs):
	# 	# -- Identify points that fall within a single cluster --
	# 	pt_lst = points_in_cluster(x_pts,y_pts,z_pts, c10, c)
	# 	print(pt_lst.shape)

	# 	# -- Find average of point list --
	# 	img_data = cluster_avg(pt_lst)
	# 	img = toimage(img_data) 
	# 	# imsave('images/space/cluster_' + str(c) +'.jpg', img)

	


	# === (3) Analysis single feature columns 3D space ===
	
	# -- Label the datapoint clusters --
	cluster_labels = []
	
	for d in range(len(x_data)):
		a_point = [x_data[d], y_data[d], z_data[d]]
		idx = closest_cluster(a_point, c10)

		cluster_labels.append(idx)


	zeros = np.zeros(len(x_data))
	for c in range(no_clstrs):
		# -- Decide origin through centering by average (alt: middle of range)
		pass

	# -- Identify points that fall within a single cluster --
	


	# === (4) Highligh difference in images ===
	input_data = pd.read_csv("../data/working-data/input-data.csv", index_col = False, header = None).values
	output_data = pd.read_csv("../data/working-data/output-data.csv", index_col = False, header = None).values

	img_size = int(math.sqrt(output_data.shape[1]))
	no_samples = output_data.shape[0]
	input_data = np.reshape(input_data, (no_samples, img_size,img_size)) 
	output_data = np.reshape(output_data, (no_samples, img_size,img_size)) 

	highlight_data = np.empty((no_samples, img_size, img_size, 3))

	diff_threshold = 80  # Change for more colouring

	for s in range(no_samples):
		for p_x in range(img_size):
			for p_y in range(img_size):
				gray_val = output_data[s][p_x][p_y]
				comp = input_data[s][p_x][p_y]

				if (abs(gray_val-comp) > diff_threshold):
					# if gray_val < 100:
					# 	red_col = gray_val

					if gray_val + 100 > 255:
						red_col = 255
					else:
						# red_col = gray_val + 100
						red_col = 255
					highlight_data[s][p_x][p_y] = [red_col,gray_val,gray_val]
				
				else:
					highlight_data[s][p_x][p_y] = [gray_val,gray_val,gray_val]
		break
		# print(highlight_data[s].flatten().shape)

	# im = Image.open("test.png") 

	# test = np.asarray(im)

	# data = np.reshape(generate_img, (img_size,img_size))