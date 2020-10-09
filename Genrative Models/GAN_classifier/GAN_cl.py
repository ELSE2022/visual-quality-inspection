

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
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



# custom activation function
def custom_activation(output):
	logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
	result = logexpsum / (logexpsum + 1.0)
	return result
 
# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(256,256,1), n_classes=3):
	# image input
	in_image = Input(shape=in_shape)
	# downsample
	fe = Conv2D(64, (4, 4), strides=(2,2), padding='same')(in_image)
	fe = LeakyReLU(0.2)(fe)
	fe = Dropout(0.3)(fe)
	fe = Conv2D(128, (4, 4), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(0.2)(fe)
	fe = Dropout(0.3)(fe)
	fe = Conv2D(256, (4, 4), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(0.2)(fe)
	fe = Dropout(0.3)(fe)
	fe = Conv2D(512, (4, 4), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(0.2)(fe)
	fe = Dropout(0.3)(fe)
	fe = Conv2D(1024, (4, 4), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(0.2)(fe)
	fe = Dropout(0.3)(fe)
	fe = Flatten()(fe)
	fe = Dropout(0.3)(fe)
	# output layer nodes
	fe = Dense(n_classes)(fe)
	# supervised output
	c_out_layer = Activation('softmax')(fe)
	# define and compile supervised discriminator model
	c_model = Model(in_image, c_out_layer)
	c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
	# unsupervised output
	d_out_layer = Lambda(custom_activation)(fe)
	# define and compile unsupervised discriminator model
	d_model = Model(in_image, d_out_layer)
	d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5))
	return d_model, c_model
 
# define the standalone generator model
def define_generator(latent_dim):
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 4x4 image
	n_nodes = 1024 * 8 * 8
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(0.2)(gen)
	gen = Reshape((8, 8, 1024))(gen)
	# upsample to 8x8
	gen = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(0.2)(gen)
	gen = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(0.2)(gen)
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(0.2)(gen)
	gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(0.2)(gen)
	out_layer = Conv2DTranspose(1, (4,4), strides=(2,2), activation='tanh', padding='same')(gen)
	model = Model(in_lat, out_layer)

	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect image output from generator as input to discriminator
	gan_output = d_model(g_model.output)
	# define gan model as taking noise and outputting a classification
	model = Model(g_model.input, gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model



# helper function to load images in memory
def load_image(filename, size=(128,128)):
    # load and resize the image
    pixels = load_img(filename, color_mode='grayscale', target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    pixels = pixels.astype('float32')
    return pixels

 
# select a supervised subset of the dataset, ensures classes are balanced
def select_supervised_samples(dataset, n_samples=90, n_classes=3):
	X, y = dataset
	X_list, y_list = list(), list()
	n_per_class = int(n_samples / n_classes)
	for i in range(n_classes):
		# get all images for this class
		X_with_class = X[y == i]
		# choose random instances
		ix = randint(0, len(X_with_class), n_per_class)
		# add to list
		[X_list.append(X_with_class[j]) for j in ix]
		[y_list.append(i) for j in ix]
	return asarray(X_list), asarray(y_list)
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	z_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = z_input.reshape(n_samples, latent_dim)
	return z_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict(z_input)
	# create class labels
	y = zeros((n_samples, 1))
	return images, y
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(9):
		# define subplot
		pyplot.subplot(3, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray')
	# save plot to file
	if not os.path.isdir('img_out/'):
		os.makedirs('img_out/')
	filename1 = 'img_out/generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1, dpi=300)
	pyplot.close()
	# evaluate the classifier model
	X, y = dataset
	_, acc = c_model.evaluate(X, y, verbose=0)
	print('Classifier Accuracy: %.3f%%' % (acc * 100))
	# save the generator model
	if not os.path.isdir('models/'):
		os.makedirs('models/')
	filename2 = 'models/g_model_%04d.h5' % (step+1)
	g_model.save(filename2)
	# save the classifier model
	filename3 = 'models/c_model_%04d.h5' % (step+1)
	c_model.save(filename3)
	print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))
 
# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=80000, n_batch=100):
	# select supervised dataset
	X_sup, y_sup = select_supervised_samples(dataset)
	print(X_sup.shape, y_sup.shape)
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
	# manually enumerate epochs
	for i in range(n_steps):
		# update supervised discriminator (c)
		[Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
		c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
		# update unsupervised discriminator (d)
		[X_real, _], y_real = generate_real_samples(dataset, half_batch)
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# update generator (g)
		X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		# summarize loss on this batch
		print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
		# evaluate the model performance every so often
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, c_model, latent_dim, dataset)
 

