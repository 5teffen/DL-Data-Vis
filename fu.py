import pyvoro
import math
import numpy as np
import pandas as pd

output_path = "Voronoi-Results/Voronoi-10.csv"

c2 = [[-1.97812533, -0.5059073, 4.6427598 ], [-3.10351944, -1.38047385, -1.28884971]]
c3 = [[-1.51348925, 2.09792948, 3.74644947],
 [-3.10579491, -1.4141202, -1.39624393],
 [-2.75118375, -4.19366789, 5.4692688 ]]
c4 = [[-3.21736574, 0.53843445, -1.50921881],
 [-1.39001787, 2.0560472,  3.91988993],
 [-2.78839064, -4.13620663, 5.76722288],
 [-2.96466017, -3.81424499, -0.88710201]]
c5 = [[-2.96918321, -3.85311866, -0.95209205],
 [-3.21269917, 0.56020826, -1.58994877],
 [-1.03266132, 4.57616091, 3.51084089],
 [-3.01424146, -4.84126282, 5.86243629],
 [-1.72918916, 0.16383761, 4.27012444]]
c6 = [[-2.86832213, -3.96254969, -0.81997001],
 [-2.01004314, 0.85829824, -0.14097381],
 [-1.02624547, 4.61557484, 3.59954095],
 [-3.08735681, -4.97374821, 5.80835915],
 [-1.65317512, 0.03919309, 4.72932291],
 [-4.51227474, -0.0338248, -2.59725523]]
c7 = [[-2.00501823, 0.5371353,  3.31690598],
 [-1.52806878, -1.40567279, 6.64615965],
 [-5.0377779, -0.17963606, -2.24186373],
 [-3.32976031, -5.45691156, 5.4157815 ],
 [-0.92226231, 4.79599476, 3.63472128],
 [-2.85422778, -3.93905997, -0.86515296],
 [-1.8382163,  0.92167991, -1.35560358]]
c8 = [[-1.97267497, 0.58433563, 3.33174658],
 [-1.52272093, 0.67971474, -1.57996166],
 [-2.69884872, -4.23658085, 0.07107008],
 [-1.52802014, -1.37688732, 6.57660913],
 [-4.9360733,  0.78364879, -1.36425078],
 [-3.37611151, -5.50906849, 5.68369818],
 [-0.90776527, 4.80811691, 3.63259983],
 [-3.77526522, -2.72540498, -2.97738361]]
c9 = [[-2.855407, -4.07619572, -1.05989254],
 [-2.9432807, -2.86026168, 3.26400161],
 [-0.81382704, 5.01943922, 3.65171123],
 [-4.96780539, -0.51076841, -2.61260366],
 [-1.48825264, 0.92708844, 4.02078295],
 [-3.41637635, -6.09821415, 6.09797764],
 [-1.45137501, -1.79077685, 7.29617071],
 [-1.39675045, 0.54850417, -1.72024214],
 [-3.61866617, 1.63762617, 0.68309879]]
c10 = [[-2.65846443, -4.3750658, -0.30079758],
 [-1.3102504,  0.64344639, 5.00166893],
 [-1.44725728, 0.59969312, -1.72995937],
 [-3.30400348, -6.46256876, 5.76951456],
 [-3.80942202, -2.68622732, -3.01234913],
 [-5.01218081, 0.72674102, -1.56736767],
 [-0.88860941, 5.09613943, 3.63429403],
 [-2.24211264, 1.26631832, 2.10446405],
 [-3.03785181, -2.75197911, 3.46823692],
 [-2.00708723, -2.70808792, 7.60714483]]


points = [[-1.51348925, 2.09792948, 3.74644947],
[-3.10579491, -1.4141202, -1.39624393],
[-2.75118375, -4.19366789, 5.4692688 ]]

range_x = (-10.494617, 2.9902372)
range_y = (-12.401706, 12.255847)
range_z = (-7.081162, 12.459479)

voronoi_list = pyvoro.compute_voronoi(c10,[range_x, range_y, range_z],1.0)

cluster = voronoi_list


def point_line(p1,p2,label,density=2):
	delta_x = p2[0]-p1[0]
	delta_y = p2[1]-p1[1]
	delta_z = p2[2]-p1[2]
	
	length = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

	x_points = np.linspace(p1[0], p2[0], length*density)
	y_points = np.linspace(p1[1], p2[1], length*density)
	z_points = np.linspace(p1[2], p2[2], length*density)
	
	result = []
	for i in range(len(x_points)):
		result.append([x_points[i],y_points[i],z_points[i],label])

	return result 


def construct_face(index_lst, vertice_lst, adj_lst, label, existing_pairs):
	result = []

	for p1 in index_lst:
		for p2 in adj_lst[p1]:
			if p2 in index_lst:

				if ((p1,p2) not in existing_pairs) and ((p2,p1) not in existing_pairs):
					point1 = vertice_lst[p1]
					point2 = vertice_lst[p2]

					result += point_line(point1,point2,label)

					existing_pairs.append((p1,p2))

					

	return result, existing_pairs


cluster_no = 0
all_faces = []
for cluster in voronoi_list:
	# all_faces = []
	existing_pairs = []

	vertices = cluster["vertices"]
	faces = cluster["faces"]
	neighbors = cluster["adjacency"]

	for f in faces:
		indexes = f['vertices']

		one_face, existing_pairs = construct_face(indexes, vertices, 
			neighbors, cluster_no, existing_pairs)
		
		all_faces += one_face

	cluster_no +=1

print(len(all_faces))

my_file = open(output_path, 'ab')
np.savetxt(my_file,all_faces, delimiter=',', fmt='%f')
my_file.close()



# def construct_face(index_lst, vertice_lst, label, existing_pairs):
# 	result = []

# 	p1 = 0
# 	for p2 in range(1,len(index_lst)):
# 		if ((p1,p2) not in existing_pairs) and ((p2,p1) not in existing_pairs):
# 			point1 = vertice_lst[index_lst[p1]]
# 			point2 = vertice_lst[index_lst[p2]]

# 			result += point_line(point1,point2,label)

# 			existing_pairs.append((p1,p2))

# 			p1 = p2

# 	return result, existing_pairs


