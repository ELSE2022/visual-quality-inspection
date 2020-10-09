# python GAN_Classifier_batch.py --test path/to/test --model_path path/to/model

import os
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
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
import argparse


from utils import predict

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=str, default='./input/test', help='path to test dataset')
parser.add_argument('--model_path', type=str, help='path to model')
args = parser.parse_args()

#####################################################################################
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#backend.set_session(session)
###################################################################################


image_l = 256
image_w = 256

LABELS = {0:'Ked_high', 1:'Ked_low', 2:'Slip_on'}

predictions = []
real_labels = []

path = args.test
model_path = args.model_path
# load the model
model = load_model(model_path)



for root, folders, imgs in os.walk(path):
	for folder in folders:
		temp = os.path.join(root, folder)
		files = os.listdir(temp)
		for file in files:
			img = os.path.join(root, folder, file)
			print(img)
			# real label is the folder name
			real_labels.append(folder)
			# predicted label
			pred = predict(img, LABELS, model, size=(image_l,image_w))
			predictions.append(pred)
			




result = pd.crosstab(np.array(real_labels), np.array(predictions), rownames=['actuals'], colnames=['preds'])
print('\n')
print(result)


report = classification_report(np.array(real_labels), np.array(predictions))
print('\n')
print(report)












