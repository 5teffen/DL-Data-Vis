"""
Scripts for perforing iterative k-means clustering

Functionality:
- Do this iteratively.
- Correction based on statistical analysis
"""


# -- Imports -- 
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

from PIL import Image

from sklearn.metrics import mean_squared_error

from exploration_utils import *


def avg_cluster_distance(points, centre):
	dist_lst = []
	for pt in points:
		dist = euc_dist(pt, centre)

		dist_lst.append(dist)

	avg_dist = np.average(dist_lst)
	std_dist = np.std(dist_lst)	
	
	return avg_dist, std_dist


def prune_single_cluster(points, centre, threshold):
	pruned_pts = []

	for pt in points:
		dist = euc_dist(pt, centre)

		if dist < threshold:
			pruned_pts.append(pt)

	return pruned_pts


def pruner(data, cluster_labels, centres, n=2):
	k = len(centres)

	# -- Divide points into cluster -- 
	pruned_data = []
	cluster_groups = [ [] for x in range(k)]

	for s in range(data.shape[0]):
		sample = data[s]
		idx = cluster_labels[s]

		cluster_groups[idx].append(list(sample))


	# -- Analyze individual clusters -- 
	pruned_pts = []
	for c in range(k):
		points = cluster_groups[c]

		avg_d, std_d = avg_cluster_distance(points, centres[c])

		threshold = avg_d + n*std_d   #Change this for increased pruning

		pruned_pts += prune_single_cluster(points, centres[c], threshold)

	return pruned_pts



# ==== Main Loop ====
if __name__ == '__main__':

	# -- Data imports -- 
	# input_data = pd.read_csv("../data/working-data/input-data.csv", index_col = False, header = None).values
	# output_data = pd.read_csv("../data/working-data/output-data.csv", index_col = False, header = None).values
	code_data = pd.read_csv("../data/working-data/reduced-data.csv", index_col = False, header = None).values
	# labels = pd.read_csv("../data/working-data/label-data.csv", index_col = False, header = None).values


	# -- Parameters -- 
	k = 10   # Number of clusters


	# -- Perform initial k-means -- 
	np.random.seed(0) 
	kmeans = KMeans(n_clusters=k ,n_init=10)
	kmeans.fit_transform(code_data)
	cluster_labels = kmeans.labels_
	centres = kmeans.cluster_centers_
	print(centres)
	

	pruned_pts = pruner(code_data,cluster_labels,centres, 1)


	# -- Perform second k-means --
	np.random.seed(0)
	kmeans = KMeans(n_clusters=k ,n_init=10)
	kmeans.fit_transform(pruned_pts)
	cluster_labels_new = kmeans.labels_
	centres_new = kmeans.cluster_centers_

	print(centres_new)


	# --- Format print operation for Centres ---




	# =============== Visualization Component ==============

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# # # ax.set_xlim3d(-1, 1)
	# # # ax.set_ylim3d(-1, 1)
	# # # ax.set_zlim3d(-1, 1)
	# plt.axis('off')
	# ax.set_aspect('equal')


	# # --- Selecting clusters to draw original ---
	# desired_1 = 5
	# desired_2 = 3
	# desired_3 = 7
	# des_pt_1 = []
	# des_pt_2 = []
	# des_pt_3 = []

	# for l in range(len(cluster_labels)):
	# 	lab = cluster_labels[l]
	# 	if lab == desired_1:
	# 		des_pt_1.append(code_data[l])
	# 	if lab == desired_2:
	# 		des_pt_2.append(code_data[l])
	# 	if lab == desired_3:
	# 		des_pt_3.append(code_data[l])


	# des_pt_1 = np.array(des_pt_1)
	# des_pt_2 = np.array(des_pt_2)
	# des_pt_3 = np.array(des_pt_3)
	# ax.scatter3D(des_pt_1[:,0], des_pt_1[:,1], des_pt_1[:,2], c='#66c2a5', s=15)
	# ax.scatter3D(des_pt_2[:,0], des_pt_2[:,1], des_pt_2[:,2], c='#fc8d62', s=15)
	# ax.scatter3D(des_pt_3[:,0], des_pt_3[:,1], des_pt_3[:,2], c='#8da0cb', s=15)
	# # plt.show()



	# # --- Selecting clusters to draw new---
	# desired_1_new = 2
	# desired_2_new = 4
	# desired_3_new = 7
	# des_pt_1_new = []
	# des_pt_2_new = []
	# des_pt_3_new = []

	# for l in range(len(cluster_labels_new)):
	# 	lab = cluster_labels_new[l]
	# 	if lab == desired_1_new:
	# 		des_pt_1_new.append(pruned_pts[l])
	# 	if lab == desired_2_new:
	# 		des_pt_2_new.append(pruned_pts[l])
	# 	if lab == desired_3_new:
	# 		des_pt_3_new.append(pruned_pts[l])

	# des_pt_1_new = np.array(des_pt_1_new)
	# des_pt_2_new = np.array(des_pt_2_new)
	# des_pt_3_new = np.array(des_pt_3_new)
	# # ax.scatter3D(des_pt_1_new[:,0], des_pt_1_new[:,1], des_pt_1_new[:,2], c='#66c2a5', s=15)
	# # ax.scatter3D(des_pt_2_new[:,0], des_pt_2_new[:,1], des_pt_2_new[:,2], c='#fc8d62', s=15)
	# # ax.scatter3D(des_pt_3_new[:,0], des_pt_3_new[:,1], des_pt_3_new[:,2], c='#8da0cb', s=15)
	# # plt.show()
