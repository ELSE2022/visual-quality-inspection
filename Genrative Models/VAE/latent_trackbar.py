# python latent_trackbar.py --weights path/to/dec_weight.h5

import cv2
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import argparse

from vae import VAE_model256X



#######################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, help='path to decoder model weights')
args = parser.parse_args()
###############################################################################################################

weights = args.weights




def load_samples(filename):
    data = load(filename)
    images, latents = data['arr_0'], data['arr_1']
    return images, latents


def nothing(x):
    pass



_, latents = load_samples('output/image2latent_256.npz') # path to latent space array

# dim = latents.shape[1]
dim = 20
factor = latents.max()


model = VAE_model256X()
decoder = model.decoder_model()
decoder.load_weights(weights)

print('weights correctly loaded...')

# Creating a window for later use
cv2.namedWindow('trackbar', cv2.WINDOW_NORMAL)

for i in range(dim):
    # Creating track bar
    cv2.createTrackbar('d{}'.format(i+1), 'trackbar', 25, 45, nothing)



while True:
    # get info from track bar and appy to result
    d1 = cv2.getTrackbarPos('d1', 'trackbar')
    d2 = cv2.getTrackbarPos('d2', 'trackbar')
    d3 = cv2.getTrackbarPos('d3', 'trackbar')
    d4 = cv2.getTrackbarPos('d4', 'trackbar')
    d5 = cv2.getTrackbarPos('d5', 'trackbar')
    d6 = cv2.getTrackbarPos('d6', 'trackbar')
    d7 = cv2.getTrackbarPos('d7', 'trackbar')
    d8 = cv2.getTrackbarPos('d8', 'trackbar')
    d9 = cv2.getTrackbarPos('d9', 'trackbar')
    d10 = cv2.getTrackbarPos('d10', 'trackbar')
    d11 = cv2.getTrackbarPos('d11', 'trackbar')
    d12 = cv2.getTrackbarPos('d12', 'trackbar')
    d13 = cv2.getTrackbarPos('d13', 'trackbar')
    d14 = cv2.getTrackbarPos('d14', 'trackbar')
    d15 = cv2.getTrackbarPos('d15', 'trackbar')
    d16 = cv2.getTrackbarPos('d16', 'trackbar')
    d17 = cv2.getTrackbarPos('d17', 'trackbar')
    d18 = cv2.getTrackbarPos('d18', 'trackbar')
    d19 = cv2.getTrackbarPos('d19', 'trackbar')
    d20 = cv2.getTrackbarPos('d20', 'trackbar')

    values = 4 * np.array([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,
                            d11,d12,d13,d14,d15,d16,d17,d18,d19,d20])

    values = -1 * np.cos(np.radians(values)) * factor
    print(values)

    values = np.expand_dims(values, axis=0)

    result = decoder.predict(values)[0]
    plt.axis('off')
    plt.imshow(result)
    plt.savefig('img.jpg', dpi=150)
    plt.close()
    result = cv2.imread('img.jpg')
    cv2.imshow('result', result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

