import argparse

import math
import numpy as np
import pandas as pd

from scipy.misc import toimage, imsave, imshow
import imageio
from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser()
parser.add_argument("coorinates", metavar='N', nargs='+', type=float)
args = parser.parse_args()


def euc_dist(p1,p2):
	result = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
	return result


if __name__ == "__main__":

	# === Highligh difference in image ===
	input_data = pd.read_csv("../data/working-data/input-data.csv", index_col = False, header = None).values
	output_data = pd.read_csv("../data/working-data/output-data.csv", index_col = False, header = None).values
	code_data = pd.read_csv("../data/working-data/reduced-data.csv", index_col = False, header = None).values

	img_size = int(math.sqrt(output_data.shape[1]))
	no_samples = output_data.shape[0]
	input_data = np.reshape(input_data, (no_samples, img_size,img_size)) 
	output_data = np.reshape(output_data, (no_samples, img_size,img_size)) 

	highlight_data = np.empty((img_size, img_size, 3))

	diff_threshold = 50  # Change for more colouring

	# s = 5 # Target index

	# s = int(args.coorinates[0])

	# test = [-1.842734, 1.068612, -2.443978]

	target = [args.coorinates[0], args.coorinates[1], args.coorinates[2]]

	# --- Identify the index using code data --- 
	min_dist = 1000
	min_i = 0
	for i in range(code_data.shape[0]):
		dist = euc_dist(target,code_data[i])

		if dist < min_dist:
			min_dist = dist
			min_i = i


	s = min_i

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
				highlight_data[p_x][p_y] = [red_col,gray_val,gray_val]
			
			else:
				highlight_data[p_x][p_y] = [gray_val,gray_val,gray_val]
	
	im = toimage(highlight_data)
	im.show()
	
