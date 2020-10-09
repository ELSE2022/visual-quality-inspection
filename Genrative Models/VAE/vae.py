


import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import os
import cv2
from numpy import random as nr
import math
import numpy as np

from keras import layers
from keras.layers import Input, Dense, Conv2DTranspose, Flatten, Reshape, Conv2D, Concatenate, Embedding, Dropout
from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import metrics
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
#from sklearn.metrics import mean_squared_error as MSE

from utils import combine_images

import warnings
warnings.filterwarnings('ignore')




class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a PAA."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon



class VAE_model256X():
    def __init__(self, latent_dim=20, intermediate_dim=100, original_shape=(256,256,3)):
        self.latent_dim = latent_dim
        self.original_shape = original_shape
        self.intermediate_dim = intermediate_dim
        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()
        self.vae = self._decoder_on_encoder()

    def _kl_reconstruction_loss(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * np.prod(self.original_shape)

        # KL divergence loss
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        print(reconstruction_loss, kl_loss)
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)


    def encoder_model(self):
        self.encoder_inputs = Input(shape=self.original_shape)
        x = Conv2D(64, (4, 4), activation="relu", strides=(2, 2), padding="same")(self.encoder_inputs)
        x = Conv2D(128, (4, 4), activation="relu", strides=(2, 2), padding="same")(x)
        x = Conv2D(256, (4, 4), activation="relu", strides=(2, 2), padding="same")(x)
        x = Dropout(0.2)(x)
        x = Conv2D(512, (4, 4), activation="relu", strides=(2, 2), padding="same")(x)
        x = Dropout(0.2)(x)
        x = Conv2D(1024, (4, 4), activation="relu", strides=(2, 2), padding="same")(x)
        x = Dropout(0.2)(x)
        x = Conv2D(1024, (4, 4), activation="relu", strides=(2, 2), padding="same")(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(self.intermediate_dim, activation ='relu')(x)

        self.z_mean = Dense(self.latent_dim, name="z_mean")(x)
        self.z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        self.z = Sampling()([self.z_mean, self.z_log_var])
        encoder = Model(inputs=self.encoder_inputs, outputs=[self.z_mean, self.z_log_var, self.z], name="encoder")
        encoder.summary()

        return encoder

    def decoder_model(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(4 * 4 * 1024, activation="relu")(latent_inputs)
        x = Reshape((4, 4, 1024))(x)
        x = Conv2DTranspose(1024, (4,4), strides=(2,2), activation="relu", padding='same')(x)
        x = Conv2DTranspose(512, (4,4), strides=(2,2), activation="relu", padding='same')(x)
        x = Conv2DTranspose(256, (4,4), strides=(2,2), activation="relu", padding='same')(x)
        x = Conv2DTranspose(128, (4,4), strides=(2,2), activation="relu", padding='same')(x)
        x = Conv2DTranspose(64, (4,4), strides=(2,2), activation="relu", padding='same')(x)
        decoder_outputs = Conv2DTranspose(3, (4,4), strides=(2,2), activation='sigmoid', padding='same')(x)
        decoder = Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
        decoder.summary()

        return decoder

    def _decoder_on_encoder(self):
        vae_outputs = self.decoder(self.encoder(self.encoder_inputs)[2])
        vae = Model(self.encoder_inputs, vae_outputs, name='VAE')
        vae.compile(optimizer='adam', loss=self._kl_reconstruction_loss)
        vae.summary()
        return vae

    def generate_samples(self, images, n_samples):
        # choose random instances
        ix = nr.randint(0, images.shape[0], n_samples)
        # select images and labels
        X = images[ix]
        return X


    def train(self, images, n_epochs=80000, n_batch=24):
        
        if not os.path.isdir('output/VAE_models'):
            os.makedirs('output/VAE_models')

        if not os.path.isdir('output/img_out'):
            os.makedirs('output/img_out')

        bat_per_epo = int(images.shape[0] / n_batch)

        vae = self._decoder_on_encoder()
        prefix = 0

        for epoch in range(n_epochs):
            prefix += 1
            for iters in range(bat_per_epo):
                samples = self.generate_samples(images, n_batch)
                loss = vae.train_on_batch(samples, samples)
                generated_images = vae.predict(samples)

                if (epoch % 2 == 0) and (iters == bat_per_epo-1):
                    n_samples = 3
                    for i in range(n_samples):
                        plt.subplot(2, n_samples, 1+i)
                        plt.axis('off')
                        plt.imshow(samples[i])

                    for i in range(n_samples):
                        plt.subplot(2, n_samples, 1+n_samples+i)
                        plt.axis('off')
                        plt.imshow(generated_images[i])
                    plt.savefig('output/img_out/{}_output.jpg'.format(epoch+1), dpi=300)
                    plt.close()
                print('epoch> %d, %d/%d, loss=%.3f' % (epoch+1, iters+1, bat_per_epo, loss))

            # save models each 10 epochs
            if (epoch + 1) % 10 == 0:
                vae_model_path = 'output/VAE_models/{}_vae_model_256x.h5'.format(epoch+1)
                enc_model_path = 'output/VAE_models/{}_enc_model_256x.h5'.format(epoch+1)
                dec_model_path = 'output/VAE_models/{}_dec_model_256x.h5'.format(epoch+1)
                vae.save_weights(vae_model_path, True)
                self.encoder.save_weights(enc_model_path, True)
                self.decoder.save_weights(dec_model_path, True)









class cVAE_model128X():
    def __init__(self, n_classes, latent_dim=20, intermediate_dim=100, original_shape=(128,128,3)):
        self.latent_dim = latent_dim
        self.original_shape = original_shape
        self.intermediate_dim = intermediate_dim
        self.n_classes = n_classes
        self.encoder = self.encoder_model()
        self.decoder = self.decoder_model()
        self.vae = self._decoder_on_encoder()

    def _kl_reconstruction_loss(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * np.prod(self.original_shape)

        # KL divergence loss
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        print(reconstruction_loss, kl_loss)
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)


    def encoder_model(self):
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self.n_classes, 50)(in_label)
        # scale up to image dimensions with linear activation
        n_nodes = np.prod(self.original_shape)
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape(self.original_shape)(li)

        # image input
        in_image = Input(shape=self.original_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, li])
        x = Conv2D(64, (4, 4), activation="relu", strides=(2, 2), padding="same")(merge)
        x = Conv2D(128, (4, 4), activation="relu", strides=(2, 2), padding="same")(x)
        x = Conv2D(256, (4, 4), activation="relu", strides=(2, 2), padding="same")(x)
        x = Conv2D(512, (4, 4), activation="relu", strides=(2, 2), padding="same")(x)
        x = Conv2D(1024, (4, 4), activation="relu", strides=(2, 2), padding="same")(x)
        x = Flatten()(x)
        x = Dense(self.intermediate_dim, activation ='relu')(x)

        self.z_mean = Dense(self.latent_dim, name="z_mean")(x)
        self.z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        self.z = Sampling()([self.z_mean, self.z_log_var])
        encoder = Model(inputs=[in_image, in_label], outputs=[self.z_mean, self.z_log_var, self.z], name="encoder")
        encoder.summary()

        return encoder

    def decoder_model(self):
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self.n_classes, 50)(in_label)
        # linear multiplication
        n_nodes = 4 * 4
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((4, 4, 1))(li)
        # decoder input
        latent_inputs = Input(shape=(self.latent_dim,))
        gen = Dense(4 * 4 * 1024, activation="relu")(latent_inputs)
        gen = Reshape((4, 4, 1024))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        x = Conv2DTranspose(512, (4,4), strides=(2,2), activation="relu", padding='same')(merge)
        x = Conv2DTranspose(256, (4,4), strides=(2,2), activation="relu", padding='same')(x)
        x = Conv2DTranspose(128, (4,4), strides=(2,2), activation="relu", padding='same')(x)
        x = Conv2DTranspose(64, (4,4), strides=(2,2), activation="relu", padding='same')(x)
        decoder_outputs = Conv2DTranspose(3, (4,4), strides=(2,2), activation='sigmoid', padding='same')(x)
        decoder = Model(inputs=[latent_inputs, in_label], outputs=decoder_outputs, name="decoder")
        decoder.summary()

        return decoder

    def _decoder_on_encoder(self):
        in_img, in_label = self.encoder.input
        encoder_z_out = self.encoder.output[2]
        vae_outputs = self.decoder([encoder_z_out, in_label])
        vae = Model([in_img, in_label], vae_outputs, name='VAE')
        vae.compile(optimizer='adam', loss=self._kl_reconstruction_loss)
        vae.summary()
        return vae

    def generate_samples(self, dataset, n_samples):
        images, labels = dataset
        # choose random instances
        ix = nr.randint(0, images.shape[0], n_samples)
        # select images and labels
        X, labels = images[ix], labels[ix]
        return X, labels


    def train(self, dataset, n_epochs=80000, n_batch=24):
        
        if not os.path.isdir('output/cVAE_models'):
            os.makedirs('output/cVAE_models')

        bat_per_epo = int(dataset[0].shape[0] / n_batch)

        vae = self._decoder_on_encoder()

        for epoch in range(n_epochs):
            for iters in range(bat_per_epo):
                samples, labels = self.generate_samples(dataset, n_batch)
                loss = vae.train_on_batch([samples, labels], samples)
                generated_images = vae.predict([samples, labels])

                if (epoch % 2 == 0) and (iters== bat_per_epo-1):
                    n_samples = 3
                    for i in range(n_samples):
                        plt.subplot(2, n_samples, 1+i)
                        plt.axis('off')
                        plt.imshow(samples[i])

                    for i in range(n_samples):
                        plt.subplot(2, n_samples, 1+n_samples+i)
                        plt.axis('off')
                        plt.imshow(generated_images[i])
                    plt.savefig('output/img_out/{}_output.jpg'.format(epoch+1), dpi=300)
                    plt.close()

                print('epoch> %d, %d/%d, loss=%.3f' % (epoch+1, iters+1, bat_per_epo, loss))

            # save models each 10 epochs
            if (epoch + 1) % 10 == 0:
                vae_model_path = 'output/cVAE_models/{}_vae_cmodel_128x.h5'.format(epoch+1)
                enc_model_path = 'output/cVAE_models/{}_enc_cmodel_128x.h5'.format(epoch+1)
                dec_model_path = 'output/cVAE_models/{}_dec_cmodel_128x.h5'.format(epoch+1)
                vae.save_weights(vae_model_path, True)
                self.encoder.save_weights(enc_model_path, True)
                self.decoder.save_weights(dec_model_path, True)





