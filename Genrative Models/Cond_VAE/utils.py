

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import os
import cv2
from numpy import random as nr
import math


from keras.preprocessing.image import load_img, img_to_array



def combine_images(generated_images):
    ### combine images for visualization
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image



# helper function to load images in memory
def load_image(filename, size=(128,128)):
    # load and resize the image
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [0,1]
    pixels = pixels / 255.
    return pixels




