
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
parser.add_argument('--directory', type=str, default='./input', help='path to dataset')
parser.add_argument('--weights', type=str, help='path to model weights')
args = parser.parse_args()
###############################################################################################################

weights = args.weights
directory = args.directory



images = []
labels = []

for root, folders, files in os.walk(directory):
    for folder in folders:
        temp = os.path.join(root, folder)
        img_names = os.listdir(temp)
        for img_name in img_names:
            img_path = os.path.join(root, folder, img_name)
            image = load_image(img_path, size=(128,128))
            images.append(image)
            labels.append(folder)



encode_label = {}  # to encode the labels in range [0, n_classes-1]
idx = 0
for value in labels:
    if value not in encode_label.keys():
        encode_label[value] = idx
        idx += 1


n_classes = len(encode_label)

labels_new = np.array([encode_label[i] for i in labels])
images = np.asarray(images)

# initialize model and load weights

model = cVAE_model128X(n_classes=n_classes)
encoder = model.encoder_model()
encoder.load_weights(weights)

_, _, latents = encoder.predict([images, labels_new])

print(images.shape, latents.shape)

# save as compressed numpy array
if not os.path.isdir('output/'):
	os.makedirs('output/')
filename = 'output/image_label2latent_128.npz'
savez_compressed(filename, images, labels_new, latents)
print('Saved data: ', filename)


