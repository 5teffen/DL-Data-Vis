import pyvoro
import math
import numpy as np
import pandas as pd

output_path = "test3.csv"

points = [[-1.51348925, 2.09792948, 3.74644947],
[-3.10579491, -1.4141202, -1.39624393],
[-2.75118375, -4.19366789, 5.4692688 ]]

range_x = (-10.494617, 2.9902372)
range_y = (-12.401706, 12.255847)
range_z = (-7.081162, 12.459479)

voronoi_list = pyvoro.compute_voronoi(points,[range_x, range_y, range_z],1.0)

cluster = voronoi_list[1]

def point_line(p1,p2,label,density=3):
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




def point_cloud(vertices, equation, density = 2):

	a,b,c,d = equation

	# max_x = np.amax(border_lines[:,0])
	# min_x = np.amin(border_lines[:,0])
	# max_y = np.amax(border_lines[:,1])
	# min_y = np.amin(border_lines[:,1])
	# max_z = np.amax(border_lines[:,2])
	# min_z = np.amin(border_lines[:,2])

	# max_x, min_x = (-10.494617, 2.9902372)
	# max_y, min_y = (-12.401706, 12.255847)
	# max_z, min_z = (-7.081162, 12.459479)


	delta_x = max_x - min_x
	delta_y = max_y - min_y
	delta_z = max_z - min_z
	
	length = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

	x_points = np.linspace(min_x, max_x, length*density)
	y_points = np.linspace(min_y, max_y, length*density)
	z_points = np.linspace(min_z, max_z, length*density)

	all_count = 0
	good_count = 0
	bad_count = 0

	for x in x_points:
		for y in y_points:
			for z in z_points:
				test = np.round((a*x + b*y + c+z +d),0)
				if test == 0:
					greater = False
					smaller = False
					for bor in border_lines:
						if (bor[0] <= x and bor[1] <= y and bor[2] <= z):
							smaller = True
						elif (bor[0] >= x and bor[1] >= y and bor[2] >= z):
							greater = True

					if greater and smaller:
						good_count += 1
				
				else:
					bad_count +=1

	
	print("GOOD",good_count)
	print("BAD",bad_count)



def equation_plane(p1,p2,p3):  
	a1 = p2[0] - p1[0]
	b1 = p2[1] - p1[1]
	c1 = p2[2] - p1[2] 
	a2 = p3[0] - p1[0] 
	b2 = p3[1] - p1[1] 
	c2 = p3[2] - p1[2] 
	a = b1 * c2 - b2 * c1 
	b = a2 * c1 - a1 * c2 
	c = a1 * b2 - b1 * a2 
	d = (- a * p1[0] - b * p1[1] - c * p1[2])
	# p1 = np.array(p1)
	# p2 = np.array(p2)
	# p3 = np.array(p3)
	# v1 = p3 - p1
	# v2 = p2 - p1
	# cp = np.cross(v1, v2)
	# a, b, c = cp
	# d = np.dot(cp, p3)

	return [a,b,c,d]



def test_colinear(p1,p2,p3,test): 
	x1, y1, z1 = p1
	x2, y2, z2 = p2
	x3, y3, z3 = p3
	x, y, z = test 
	a1 = x2 - x1 
	b1 = y2 - y1 
	c1 = z2 - z1 
	a2 = x3 - x1 
	b2 = y3 - y1 
	c2 = z3 - z1 
	a = b1 * c2 - b2 * c1 
	b = a2 * c1 - a1 * c2 
	c = a1 * b2 - b1 * a2 
	d = (- a * x1 - b * y1 - c * z1) 

	if(a * x + b * y + c * z + d == 0): 
		return True
	else: 
		return False
      

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
		# print((vertices[indexes[0]],vertices[indexes[1]],vertices[indexes[2]]))
		eq = equation_plane(vertices[indexes[0]],
									vertices[indexes[1]],
										vertices[indexes[2]])

		one_face, existing_pairs = construct_face(indexes, vertices, 
			neighbors, cluster_no, existing_pairs)

		point_cloud(np.array(one_face), eq)
		
		all_faces += one_face

	exit()
	cluster_no +=1

# print(len(all_faces))

# my_file = open(output_path, 'ab')
# np.savetxt(my_file,all_faces, delimiter=',', fmt='%f')
# my_file.close()

