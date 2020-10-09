

## python VAE_cond_main.py --directory path/to/images --epochs <num of epochs> --batch_size <batch size>

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import os
import cv2
from numpy import random as nr
import math
import argparse
import json

from keras import layers
from keras.layers import Input, Dense, Conv2DTranspose, Flatten, Reshape, Conv2D, Concatenate, Embedding
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import metrics
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import mean_squared_error as MSE

from utils import load_image
from vae import cVAE_model128X

import warnings
warnings.filterwarnings('ignore')



#######################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='./input', help='path to dataset')
parser.add_argument('--epochs', type=int, default=600, help='epochs to train the model')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
args = parser.parse_args()
###############################################################################################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

############################################################################################################

batch_size = args.batch_size
epochs = args.epochs
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



if not os.path.isdir('label_json/'):
	os.makedirs('label_json/')

with open('label_json/encode_labels.json', 'w') as f:
	json.dump(encode_label, f, indent=4)



n_classes = len(encode_label)

labels_new = np.array([encode_label[i] for i in labels])
images = np.asarray(images)

dataset = [images, labels_new]



# initialize and train the model

model = cVAE_model128X(n_classes=n_classes)
model.train(dataset, n_epochs=epochs, n_batch=batch_size)






