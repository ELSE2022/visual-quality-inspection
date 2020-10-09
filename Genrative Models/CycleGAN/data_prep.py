
# python data_prep.py

import os
from os import listdir
import argparse
from numpy import asarray, savez_compressed, vstack
from utils import load_images

###########################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./input/real_and_synth', help='path to the images main folder')
parser.add_argument('--resize', type=int, nargs='+', default=[256, 256], help='resize the input images')
args = parser.parse_args()
print(args)

###########################################################################################################
 

 
# dataset path
path = args.path
# resize
size = tuple(args.resize)

# load dataset A
dataA = load_images(os.path.join(path, 'real/'), size=size)
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB = load_images(os.path.join(path, 'synth/'), size=size)
print('Loaded dataB: ', dataB.shape)

# save as compressed numpy array
if not os.path.isdir('input/'):
	os.makedirs('input/')
filename = 'input/real2synth_256.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)




