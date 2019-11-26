import pyvoro
import math
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull

output_path = "plswork2.csv"

points = [[-1.51348925, 2.09792948, 3.74644947],
[-3.10579491, -1.4141202, -1.39624393],
[-2.75118375, -4.19366789, 5.4692688 ]]

range_x = (-10.494617, 2.9902372)
range_y = (-12.401706, 12.255847)
range_z = (-7.081162, 12.459479)

voronoi_list = pyvoro.compute_voronoi(points,[range_x, range_y, range_z],1.0)

def point_line(p1,p2,label,density=5):
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

def equation_plane(p1,p2,p3):  
	p1 = np.array(p1)
	p2 = np.array(p2)
	p3 = np.array(p3)
	v1 = p3 - p1
	v2 = p2 - p1
	cp = np.cross(v1, v2)
	a, b, c = cp
	d = np.dot(cp, p3)

	return [a,b,c,d]

def equal_arrays(arr1, arr2): 

	if (len(arr1) != len(arr2)): 
		return False; 
	 
	for i in range(len(arr1)): 
		if (arr1[i] != arr2[i]): 
			return False; 

	return True; 

def point_cloud(vertices, border_lines, density = 2):

	a,b,c,d = equation_plane(vertices[0],vertices[1],vertices[2])
	
	max_x, min_x = (-10.494617, 2.9902372)
	max_y, min_y = (-12.401706, 12.255847)
	max_z, min_z = (-7.081162, 12.459479)

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

	plane_points = []
	for x in x_points:
		for y in y_points:
			if c == 0:
				z = 0 
			else:
				z = np.round(((-d - a*x - b*x)/c),1)

			plane_points.append(np.round([x,y,z],0))

	# -- Check which points lie within --
	# final_plane = []
	# for p in plane_points:
	# 	x,y,z = p

	# 	greater = False
	# 	smaller = False
		
	# 	for bor in vertices:
	# 		if (bor[0] <= x or bor[1] <= y or bor[2] <= z):
	# 			smaller = True
	# 		if (bor[0] >= x or bor[1] >= y or bor[2] >= z):
	# 			greater = True

	# 	if greater and smaller:
	# 		final_plane.append(p)

	vertices = np.round(vertices,0)

	if len(vertices) <= 3:
		vertices = np.append(vertices, [vertices[0]], axis = 0)
	
	hull = ConvexHull(vertices, qhull_options="QJ")
	hull_v = [hull.points[i] for i in hull.vertices]
	
	final_plane = []

	for p in plane_points:
		new_pts = np.append(vertices, [p], axis = 0)

		test_hull = ConvexHull(new_pts, qhull_options="QJ")
		test_hull_v = [test_hull.points[i] for i in test_hull.vertices]
		
		if np.absolute(hull.area - test_hull.area) < 10:
			final_plane.append(p)


		# if np.array_equal(hull_v,test_hull_v):
		# 	final_plane.append(p)
		# else:
		# 	print(hull_v)
		# 	print(test_hull_v)

	return np.array(final_plane)



if __name__ == '__main__':
	cluster_no = 2
	all_faces = np.empty(0)
	for cluster in voronoi_list:
		existing_pairs = []

		vertices = cluster["vertices"]
		faces = cluster["faces"]
		neighbors = cluster["adjacency"]

		for f in faces:
			indexes = f['vertices'] # vertice indexes
			relevant_vertices = [vertices[i] for i in indexes]

			one_face, existing_pairs = construct_face(indexes, vertices, 
				neighbors, cluster_no, existing_pairs)

			result = point_cloud(relevant_vertices,np.array(one_face))
			if result.shape[0] != 0:
				if all_faces.shape[0] == 0:
					all_faces = result

				else:
					all_faces = np.append(all_faces,result,axis=0)


		cluster_no +=1
	

	print(all_faces.shape)
	my_file = open(output_path, 'ab')
	np.savetxt(my_file,all_faces, delimiter=',', fmt='%f')
	my_file.close()