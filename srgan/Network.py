#!/usr/bin/env python
#title           :Network.py
#description     :Architecture file(Generator and Discriminator)
#author          :Deepak Birla
#date            :2018/10/30
#usage           :from Network import Generator, Discriminator
#python_version  :3.5.4 

# Modules
import keras
import math
import tensorflow as tf
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
import Parameters
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# Residual block
def res_block_gen(model, kernel_size, filters, strides):
	gen = model
	model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
	model = BatchNormalization(momentum = 0.5)(model)
	# Using Parametric ReLU
	model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
	model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
	model = BatchNormalization(momentum = 0.5)(model)
	model = add([gen, model]) # elementwise sum
	return model
	
	
def up_sampling_block(model, upsample_size, kernel_size, filters, strides):
	# In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
	# Even we can have our own function for deconvolution (i.e one made in Utils.py)
	# model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
	model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
	# model = UpSampling2D(size = 2)(model)
	model = UpSampling2D(size = upsample_size)(model)
	model = LeakyReLU(alpha = 0.2)(model)
	return model


def discriminator_block(model, filters, kernel_size, strides):
	model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
	model = BatchNormalization(momentum = 0.5)(model)
	model = LeakyReLU(alpha = 0.2)(model)
	return model


class Generator(object):
	def __init__(self, noise_shape):
		self.noise_shape = noise_shape

	def generator(self):
		gen_input = Input(shape = self.noise_shape)
		model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
		model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
		gen_model = model
		
		# Using 16 Residual Blocks
		for index in range(16):
			model = res_block_gen(model, kernel_size = 3, filters = 64, strides = 1)
		
		model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
		model = BatchNormalization(momentum = 0.5)(model)
		model = add([gen_model, model]) # elementwise sum
		
		# Using UpSampling Blocks
		if self.noise_shape == (64, 64, 3):
			for _ in range(2):
				model = up_sampling_block(model, upsample_size=(2, 2), kernel_size=3, filters=256, strides=1)
		elif self.noise_shape == (128, 128, 3):
			model = up_sampling_block(model, upsample_size=(2, 2), kernel_size=3, filters=256, strides=1)
		elif self.noise_shape == (128, 64, 3):
			model = up_sampling_block(model, upsample_size=(2, 2), kernel_size=3, filters=256, strides=1)
			model = up_sampling_block(model, upsample_size=(1, 2), kernel_size=3, filters=256, strides=1)
    
		model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
		model = Activation('tanh')(model)
		generator_model = Model(inputs = gen_input, outputs = model)
		
		return generator_model

class Discriminator(object):
	def __init__(self, image_shape):
		self.image_shape = image_shape

	def discriminator(self):
		dis_input = Input(shape = self.image_shape)
		
		model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
		model = LeakyReLU(alpha = 0.2)(model)
		
		model = discriminator_block(model, 64, 3, 2)
		model = discriminator_block(model, 128, 3, 1)
		model = discriminator_block(model, 128, 3, 2)
		model = discriminator_block(model, 256, 3, 1)
		model = discriminator_block(model, 256, 3, 2)
		model = discriminator_block(model, 512, 3, 1)
		model = discriminator_block(model, 512, 3, 2)
		
		model = Flatten()(model)
		model = Dense(1024)(model)
		model = LeakyReLU(alpha = 0.2)(model)
	    
		model = Dense(1)(model)
		model = Activation('sigmoid')(model) 
		
		discriminator_model = Model(inputs = dis_input, outputs = model)
		
		return discriminator_model


if __name__== "__main__":
    noise_shape = (64, 64, 3)
    discriminator_shape = (256, 256, 3)
    gen = Generator(noise_shape=noise_shape)
    model = gen.generator()
    model.summary()
    
    disc = Discriminator(discriminator_shape)
    dmodel = disc.discriminator()
    dmodel.summary()