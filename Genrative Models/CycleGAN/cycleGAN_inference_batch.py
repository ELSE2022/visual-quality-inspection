
### python cycleGAN_inference_batch.py --path path/to/img.jpg --model path/to/saved_model.h5

import os
import numpy as np
from numpy import load
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import tensorflow as tf
from keras import backend as K
import cv2

import argparse
from matplotlib import pyplot

from utils import load_image

import warnings
warnings.filterwarnings('ignore')

###################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./input/test', help='images path, str')
parser.add_argument('--model', type=str, help='path to the model, str')

args = parser.parse_args()
###################################################################################

path = args.path
model = args.model



###################################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
####################################################################


# load the model
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model(model, cust)

prefix = model.split('_')[-1].split('.')[0] + '_'

output = 'output/domainB'
if not os.path.isdir(output):
	os.makedirs(output)


for root, folders, imgs in os.walk(path):
	for folder in folders:
		temp = os.path.join(root, folder)
		files = os.listdir(temp)
		for file in files:
			img = os.path.join(root, folder, file)
			# load the image
			image_src = load_image(img)
			# translate image
			image_tar = model_AtoB.predict(image_src)
			# scale from [-1,1] to [0,1]
			image_tar = (image_tar + 1) / 2.0
			temp = filename = os.path.join(output, folder)
			if not os.path.isdir(temp):
				os.makedirs(temp)

			filename = os.path.join(temp, prefix + os.path.split(img)[1])
			image = image_tar[0]
			image = (255 * image).astype(np.uint8)
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			cv2.imwrite(filename, image)
			print(filename)







