
## python image2latent.py --directory path/to/images --weights path/to/weight.h5


import matplotlib.pyplot as plt
#from scipy.stats import norm
import numpy as np
import os
import cv2
from numpy import random as nr
import math
import argparse
from numpy import asarray, savez_compressed, vstack

from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import metrics
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from utils import combine_images, load_image
from vae import VAE_model256X

import warnings
warnings.filterwarnings('ignore')



#######################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='input/images', help='path to dataset')
parser.add_argument('--weights', type=str, help='path to model weights')
args = parser.parse_args()
###############################################################################################################

weights = args.weights
directory = args.directory



img_names = os.listdir(directory)

images = []

for img_name in img_names:
    image = load_image(os.path.join(directory, img_name), size=(256,256))
    images.append(image)


images = asarray(images)

# initialize model and load weights

model = VAE_model256X()
encoder = model.encoder_model()
encoder.load_weights(weights)

_, _, latents = encoder.predict(images)

print(images.shape, latents.shape)

# save as compressed numpy array
if not os.path.isdir('output/'):
	os.makedirs('output/')
filename = 'output/image2latent_256.npz'
savez_compressed(filename, images, latents)
print('Saved data: ', filename)


