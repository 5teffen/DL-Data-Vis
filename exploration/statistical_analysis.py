"""
A qualitative analysis of the k-clusters

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
from cluster_pruning import *



def cluster_sorter(centers, labels):
	# -- Sort arbitrarily using L1 distance -- 
	l1_centers = []
	for c in range(len(centers)):
		l1 = math.sqrt((centers[c][0])**2+(centers[c][1])**2+(centers[c][2])**2)
		l1_centers.append(l1)

	print(l1_centers)


def match_cluster(original, centers, labels):
	result_c = []
	result_l = []

	conversion = {}
	
	# -- Match to closest in original -- 
	for p1 in range(len(original)):
		pt1 = original[p1]
		smallest = 10000
		i = None

		for p2 in range(len(centers)):
			pt2 = centers[p2]

			dist = euc_dist(pt1, pt2)

			if dist < smallest:
				smallest = dist
				i = p2

		result_c.append(list(centers[i]))

		conversion[str(i)] = p1

	# -- Convert labels --
	for p in range(len(labels)):
		orig = str(labels[p])
		new = conversion[orig]

		result_l.append(new)

	return np.array(result_c), np.array(result_l)



# ==== Main Loop ====
if __name__ == '__main__':

	# --- Data imports --- 
	# input_data = pd.read_csv("../data/working-data/input-data.csv", index_col = False, header = None).values
	# output_data = pd.read_csv("../data/working-data/output-data.csv", index_col = False, header = None).values
	code_data = pd.read_csv("../data/working-data/reduced-data.csv", index_col = False, header = None).values
	labels = pd.read_csv("../data/working-data/label-data.csv", index_col = False, header = None).values


	# --- Parameters --- 
	k = 10   # Number of clusters

	# --- Perform initial k-means --- 
	np.random.seed(0) 
	kmeans = KMeans(n_clusters=k ,n_init=10)
	kmeans.fit_transform(code_data)
	cluster_labels = kmeans.labels_
	centres = kmeans.cluster_centers_
	# print(centres)
	

	pruned_pts = pruner(code_data,cluster_labels,centres, 1)

	# --- Perform second k-means ---
	np.random.seed(0)
	kmeans = KMeans(n_clusters=k ,n_init=10)
	kmeans.fit_transform(pruned_pts)
	cluster_labels_new = kmeans.labels_
	centres_new = kmeans.cluster_centers_


	# ===== Statistical Analysis of Clusters ======

	# --- Before pruning ---
	centres1 = centres
	labels1 = cluster_labels
	outliers = True

	# --- Only changing centers --- 
	centres1, labels1 = match_cluster(centres1, centres_new, cluster_labels)
	outliers = True
	
	# --- After pruning ---
	# centres1, labels1 = match_cluster(centres1, centres_new, cluster_labels_new)
	# outliers = False

	total_no = len(labels1)
	cluster_count = [{"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0} for x in range(k)]  # Each dictionary signifies a cluster

	# --- Analysis with all points ---
	for p in range(code_data.shape[0]):
		pt = code_data[p]
		lab = int(labels[p][0])				# Label index
		ci = closest_cluster(pt, centres1)  # Cluster index
		
		if not outliers:
			if (np.array(pt) == np.array(pruned_pts)).all(1).any():
				cluster_count[ci][str(lab)] += 1
		else:
			# Incrament appropriate count 
			cluster_count[ci][str(lab)] += 1


	#--- Total cluster point count ---
	total_error = 0
	for cc in range(len(cluster_count)):
		cluster = cluster_count[cc]
		total = 0
		for k in cluster:
			total += cluster[k]
		
		total_error += abs(total-total_no/10)

		# --- Output values --- 
		print("Cluster: " + str(cc) + " has " + str(total) + " points.")

	print("Total Error: " + str(total_error))



	# --- Majority label count ---
	perc_lst = []
	for cc in range(len(cluster_count)):
		cluster = cluster_count[cc]
		total = 0
		for k in cluster:
			total += cluster[k]
		
		mk = max(cluster, key=cluster.get) # max key
		perc = 100*cluster[mk]/total

		perc_lst.append(perc)

		print("Cluster: " + str(cc) + " Label: " + str(mk) + " Percentage: " + str(perc))
		print(cluster)

	perc_avg = np.average(perc_lst)
	
	print("Total Percentage Average: " + str(perc_avg))
		




	# # --- Statistical Analysis of Clusters ---
	# clusters = np.reshape(clusters,(clusters.shape[0],1)).astype(int)
	# labels_data = np.reshape(labels_data,(labels_data.shape[0],1)).astype(int)

	# label_count = [{} for x in range(10)]     # Each dictionary signifies a label showing how it is distribituted
	# cluster_count = [{} for x in range(10)]   # Each dictionary signifies a cluster

	# test_file = np.append(labels_data, clusters, axis = 1)


	# # --- Analyse label distribution --- 
	# for f in range(test_file.shape[0]):
	# 	label_dic = label_count[test_file[f][0]]
	# 	cluster_no = test_file[f][1]

	# 	if cluster_no in label_dic:
	# 		label_dic[cluster_no] += 1
	# 	else:
	# 		label_dic[cluster_no] = 1

	# # print(label_count)

	# # for d in range(len(label_count)):
	# # 	dic = label_count[d]
	# # 	sorted_lst = max(dic.items(), key=operator.itemgetter(1))
	# # 	m = sorted_lst[0]   # Mosd
	# # 	s = sorted_lst[1]
	# # 	print("Digit " + str(d) + " -- Cluster " + str(m+1))


	# # --- Analyse cluster contents --- 
	# for f in range(test_file.shape[0]):
	# 	cluster_dic = cluster_count[test_file[f][1]]
	# 	label_no = test_file[f][0]

	# 	if label_no in cluster_dic:
	# 		cluster_dic[label_no] += 1
	# 	else:
	# 		cluster_dic[label_no] = 1

	# print(cluster_count)

	# # --- Detail distribution of cluster --- 
	# for d in range(len(cluster_count)):
	# 	dic = cluster_count[d]
	# 	sorted_lst = max(dic.items(), key=operator.itemgetter(1))
	# 	total_count = 0
	# 	for c in dic:
	# 		total_count += dic[c]

	# 	print(sorted_lst)
	# 	m = sorted_lst[0]   # Mosd
	# 	s = sorted_lst[1]

	# 	highest_label_count = float(dic[m])
	# 	per_of_total = round(highest_label_count/total_count*100.0)

	# 	print("Cluster " + str(d) + ": -1st Label " + str(m) + " -Percentage "  + str(per_of_total) + "% -Total Count " + str(total_count)+ " -2nd Label TBD")



