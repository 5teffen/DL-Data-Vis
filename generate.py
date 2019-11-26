import argparse
import numpy as np
import math

from scipy.misc import toimage, imsave

import torch
import torch.nn as nn
from torch.autograd import Variable
from model_linear import Autoenc

parser = argparse.ArgumentParser()
parser.add_argument("coorinates", metavar='N', nargs='+', type=float)
args = parser.parse_args()



if __name__ == "__main__":

	model_path = './full_model.pth'

	xyz = Variable(torch.FloatTensor(args.coorinates), requires_grad = False)

	mapping = Autoenc()
	mapping.load_state_dict(torch.load(model_path, map_location = 'cpu'))
	mapping.eval()

	generate_img = mapping.decoder(xyz)
	generate_img  = (generate_img.cpu().detach().numpy()*255).astype(int)
	generate_img[generate_img < 0] = 0

	img_size = int(math.sqrt(generate_img.shape[0]))
	data = np.reshape(generate_img, (img_size,img_size)) #reshape image

	img = toimage(data)  
	# img.show()       # uncomment to show image
	# imsave("test.jpg",img)  # uncomment to save image








