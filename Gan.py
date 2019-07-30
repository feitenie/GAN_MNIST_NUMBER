#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:56:12 2019

@author: feitenie
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  Conv2D,Conv2DTranspose
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

def load_data():
  K.set_image_dim_ordering('th')
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train = (X_train.astype(np.float32) - 127.5)/127.5
  X_train = X_train[:, np.newaxis, :, :]
  return X_train

#generaotr building
def G_build():
    generator = Sequential()
    generator.add(Dense(32*5 *5, input_dim = randomDim))
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((32, 5, 5)))
    generator.add(Conv2DTranspose(32, 3, strides=(2, 2), padding='valid', output_padding=None, use_bias=False))
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(64, 5, strides=(2, 2), padding='valid', output_padding=None, use_bias=False))
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(1, 4, strides=(1, 1), padding='valid', output_padding=None, use_bias=False))
    return generator

#discriminator building
def D_build():
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', input_shape=(1, 28, 28), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.0003))
    return discriminator

def plot_generated_image():
  generated_image = generator.predict(np.random.randint(0, X_train.shape[0], size=[1, randomDim]))
  dims = generated_image.shape
  plt.pcolor(generated_image.reshape(dims[2], dims[3]), cmap = plt.cm.gray)
  plt.show()

def train(epochs = 1, batch_size = 16):
  epochs = 10
  batch_count = X_train.shape[0] / batch_size

  for i in range(1, epochs+1):
      for j in range(int(batch_count)):
          #step 1.1, generate the image from noise/ normal distribution ---Terrible data
          noise_fake = np.random.normal(0, 1, size=[batch_size, randomDim])
          fake_image = generator.predict(noise_fake)
          #step 1.2, get real image from training data --- Good data
          real_image = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
    
          # mix data from step1, and label them respectively
          X = np.concatenate([real_image, fake_image])
          Y = np.zeros(2*batch_size)
          Y[:batch_size] = 0.9
    
          # Step 2, train the discriminator alone
          discriminator.trainable = True
          dloss = discriminator.train_on_batch(X, Y)
          #fit(X, Y, batch_size = batch_size , epochs = 4, shuffle = True)
    
          # Step 3, train generator AND  discriminator non-update
          noise = np.random.normal(0, 1, size=[batch_size, randomDim])
          Y = np.ones(batch_size)
          discriminator.trainable = False
          gloss = gan.train_on_batch(noise, Y)#(noise, Y, batch_size = batch_size , epochs = 4, shuffle = True)
          plot_generated_image()
      # Store loss of most recent batch from this epoch
      dLosses.append(dloss)
      gLosses.append(gloss)
      
if __name__ == "__main__":

  randomDim = 100  
  X_train = load_data()  
  generator = G_build()
  discriminator = D_build()
  ganInput = Input(shape=(randomDim,))
  discriminator.trainable = False
  x = generator(ganInput)
  ganOutput = discriminator(x)
  gan = Model(inputs=ganInput, outputs=ganOutput)
  gan.compile(loss='binary_crossentropy', optimizer = Adam(lr = 0.0003) )
  dLosses = []
  gLosses = []
  train(epochs = 1, batch_size = 16)