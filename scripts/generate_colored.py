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



if __name__ == "__main__":

	# === Highligh difference in image ===
	input_data = pd.read_csv("../data/working-data/input-data.csv", index_col = False, header = None).values
	output_data = pd.read_csv("../data/working-data/output-data.csv", index_col = False, header = None).values

	img_size = int(math.sqrt(output_data.shape[1]))
	no_samples = output_data.shape[0]
	input_data = np.reshape(input_data, (no_samples, img_size,img_size)) 
	output_data = np.reshape(output_data, (no_samples, img_size,img_size)) 

	highlight_data = np.empty((img_size, img_size, 3))

	diff_threshold = 80  # Change for more colouring

	# s = 5 # Target index

	s = int(args.coorinates[0])


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
	# im.show()
	
