#!/usr/bin/env python
#title           :Utils_model.py
#description     :Have functions to get optimizer and loss
#author          :Deepak Birla
#date            :2018/10/30
#usage           :imported in other files
#python_version  :3.5.4

from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model, Input
import tensorflow as tf
import keras
from keras import Model
import numpy as np         
from Parameters import *   


class VGGLOSS(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
        return K.mean(K.square(self.model(y_true) - self.model(y_pred)))
        
    


