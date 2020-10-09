# python GAN_cl_main.py --train path/to/train

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
import argparse
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from matplotlib import pyplot
import tensorflow as tf
from keras import backend

from GAN_cl import *

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default='./input/train', help='path to train dataset')
args = parser.parse_args()



#####################################################################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
backend.set_session(session)
###################################################################################


# load image data

label_dict = {'Ked_high':0, 'Ked_low':1, 'Slip_on':2}
images = []
labels = []



path = args.train

for root, folders, imgs in os.walk(path):
	for folder in folders:
		temp = os.path.join(root, folder)
		files = os.listdir(temp)
		for file in files:
			img = os.path.join(root, folder, file)
			pixels = load_image(img, size=(256,256))
			images.append(pixels)
			# label is the folder name
			labels.append(label_dict[folder])


images = asarray(images)
labels = asarray(labels)

print('Data informations:', images.shape, labels.shape)

dataset = [images, labels]


# size of the latent space
latent_dim = 100
# create the discriminator models
d_model, c_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)

# train model
train(g_model, d_model, c_model, gan_model, dataset, latent_dim)

