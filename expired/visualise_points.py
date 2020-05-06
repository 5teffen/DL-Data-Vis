from mpl_toolkits.mplot3d import Axes3D 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_name = "plswork2.csv"

range_x = (-10.494617, 2.9902372)
range_y = (-12.401706, 12.255847)
range_z = (-7.081162, 12.459479)

all_data = pd.read_csv(file_name,header=None).values

fig = plt.figure(figsize=(100,100))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(range_x)
ax.set_ylim(range_y)
ax.set_zlim(range_z)
# ax.set_axis_off()


colours = ['red','blue','green','orange','black','violet','cyan','gold','saddlebrown','lightpink']

for i in range(all_data.shape[0]):
	# x_val, y_val, z_val, label = all_data[i]
	# if label==2:
	# 	ax.scatter(x_val, y_val, z_val, c=colours[int(label)], s=5, marker = ('$'+str(int(label))+'$'))
	x_val, y_val, z_val= all_data[i]
	ax.scatter(x_val, y_val, z_val, c=colours[1], s=5)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# ax.legend()
plt.show()