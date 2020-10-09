
import os
import numpy as np
from numpy import expand_dims
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from keras.models import load_model
from matplotlib import pyplot
import tensorflow as tf
from keras import backend




def load_image(filename, size=(256,256)):
    # load and resize the image
    img = load_img(filename, color_mode='grayscale', target_size=size)
    # convert to numpy array
    img = img_to_array(img)
    img = (img - 127.5) / 127.5
    img = img.astype('float32')
    return img




def predict(filename, labels, c_model, size=(256,256)):
    img = load_image(filename, size)
    img = np.expand_dims(img, 0)
    # compute the predict probabilities
    prob = c_model.predict(img)
    a = np.argmax(prob)
    label = labels[a]

    return label