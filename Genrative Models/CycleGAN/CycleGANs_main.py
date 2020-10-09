

## python CycleGANs_main.py --dataset path/to/data.npz


import os
from os import listdir
from random import random
import numpy as np
from numpy import asarray, savez_compressed, vstack, load, zeros, ones
from numpy.random import randint
from matplotlib import pyplot
import cv2
import math
import argparse

from keras.models import Sequential, Model
from keras.layers import (Input, Reshape, Dense, Dropout, UpSampling2D, Conv2D,
                            Conv2DTranspose, Flatten, BatchNormalization, Activation, Concatenate)
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.utils.generic_utils import Progbar
from keras.initializers import RandomNormal
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from gans import CycleGAN
from utils import *



import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./input/real2synth_256.npz', help='path to the .npz compressed numpy array dataset')
parser.add_argument('--epochs', type=int, default=5000, help='epochs to train the model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
args = parser.parse_args()
print(args)
############################################################################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


##############################################################################################





# load image data
dataset = load_real_samples(args.dataset)
batch_size = args.batch_size
epochs = args.epochs
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]

cyclegan = CycleGAN(image_shape=image_shape)
# generator: A -> B
g_model_AtoB = cyclegan.define_generator()
# generator: B -> A
g_model_BtoA = cyclegan.define_generator()
# discriminator: A -> [real/fake]
d_model_A = cyclegan.define_discriminator()
# discriminator: B -> [real/fake]
d_model_B = cyclegan.define_discriminator()
# composite: A -> B -> [real/fake, A]
c_model_AtoB = cyclegan.define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = cyclegan.define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB)


# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, n_epochs=epochs, n_batch=batch_size)






