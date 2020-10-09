
## python VAE_main.py --directory path/to/images --epochs < num of epochs > --batch_size < batch size dim >


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
#from sklearn.metrics import mean_squared_error as MSE

from utils import combine_images, load_image
from vae import VAE_model256X

import warnings
warnings.filterwarnings('ignore')



#######################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='./input/images', help='path to dataset')
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


img_names = os.listdir(directory)

images = []

for img_name in img_names:
    image = load_image(os.path.join(directory, img_name), size=(256,256))
    images.append(image)


# initialize and train the model

model = VAE_model256X()
images = asarray(images)

model.train(images, n_epochs=epochs, n_batch=batch_size)

print('Training completed...')





