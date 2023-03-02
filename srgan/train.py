#!/usr/bin/env python
#title           :train.py
#description     :to train the model
#author          :Deepak Birla
#date            :2018/10/30
#usage           :python train.py --options
#python_version  :3.5.4 

from Network import Generator, Discriminator
import Parameters
from VggLoss import VGGLOSS
from SrganDataLoader import SrganDataLoader

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input
import keras.backend as K
from tqdm import tqdm
import numpy as np
import os
import time
import random
import argparse
import warnings
import pickle
from tensorflow.python.client import device_lib

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(42)
random.seed(42)

# Remember to change sample rate if you are having different size of images

# Combined network
def mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def get_gan_network(discriminator, shape, generator, optimizer, vggloss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vggloss, "binary_crossentropy"],
                loss_weights=[2., 1e-3],
                optimizer=optimizer)
    return gan

# def get_batch_data(lr_images, lr_label, hr_dict, rand_num):
#     lr_data = np.array(lr_images)[rand_num]
#     hr_data = []
#     for i in rand_num:
#         hr_data.append(hr_dict[lr_label[i]])
#     return lr_data, np.array(hr_data)

def get_batch_data(fall_lr, fall_lr_label, fall_hr_dict, fall_rand_nums, 
                   nonfall_lr, nonfall_lr_label, nonfall_hr_dict, nonfall_rand_nums):
    fall_lr_data = np.array(fall_lr)[fall_rand_nums]
    fall_hr_data = []
    for i in fall_rand_nums:
        fall_hr_data.append(fall_hr_dict[fall_lr_label[i]])
    # print("fall data shape >> ", fall_lr_data.shape, len(fall_hr_data), fall_hr_data[0].shape)
        
    nonfall_lr_data = np.array(nonfall_lr)[nonfall_rand_nums]
    nonfall_hr_data = []
    for i in nonfall_rand_nums:
        nonfall_hr_data.append(nonfall_hr_dict[nonfall_lr_label[i]])
    # print("nonfall data shape >> ", nonfall_lr_data.shape, len(nonfall_hr_data), nonfall_hr_data[0].shape)
        
    lr_data = np.concatenate([fall_lr_data, nonfall_lr_data], axis=0)
    hr_data = np.concatenate([fall_hr_data, nonfall_hr_data], axis=0)
    # print("batch data shape >> ", lr_data.shape, hr_data.shape)
    
    return lr_data, hr_data

# default values for all parameters are given, if want defferent values you can give via commandline
# for more info use $python train.py -h
def train(batch_size, epochs, learning_rate, load, save): 
    
    fall_lr = np.load('./training_data/' + str(Parameters.lr_sample_rate) + '_fall_lr.npy')
    fall_lr_label = np.load('./training_data/' + str(Parameters.lr_sample_rate) + '_fall_lr_label.npy')
    nonfall_lr = np.load('./training_data/' + str(Parameters.lr_sample_rate) + '_nonfall_lr.npy')
    nonfall_lr_label = np.load('./training_data/' + str(Parameters.lr_sample_rate) + '_nonfall_lr_label.npy')
    fall_hr_dict, nonfall_hr_dict = None, None
    with open('./training_data/' + str(Parameters.lr_sample_rate) + '_fall_hr_dict.pickle', 'rb') as fr:
        fall_hr_dict = pickle.load(fr)
    with open('./training_data/' + str(Parameters.lr_sample_rate) + '_nonfall_hr_dict.pickle', 'rb') as fr:
        nonfall_hr_dict = pickle.load(fr)
    print("fall data loaded, len ", len(fall_lr), len(fall_lr_label), len(fall_hr_dict)) 
    print("fall data shape ", fall_lr[0].shape, fall_hr_dict[0].shape)
    print("nonfall data loaded, len ", len(nonfall_lr), len(nonfall_lr_label), len(nonfall_hr_dict)) 
    print("nonfall data shape ", nonfall_lr[0].shape, nonfall_hr_dict[0].shape)
    
    print('\n', ':'*50)
    print('', ':'*16, "  data loaded   ", ':'*16)
    print('', ':'*50, '\n')
    
    lr_shape = Parameters.lr_image_shape
    hr_shape = Parameters.hr_image_shape
    
    vgg_loss = VGGLOSS(image_shape=hr_shape)
    # fall_batch_count = int(fall_lr.shape[0] / batch_size)
    # nonfall_batch_count = int(nonfall_lr.shape[0] / batch_size)
    batch_count = int((fall_lr.shape[0] + nonfall_lr.shape[0]) / batch_size)
    
    generator = Generator(lr_shape).generator()
    discriminator = Discriminator(hr_shape).discriminator()
    
    if load is not None:
        generator = load_model('./srgan_model/' + load + '_gen_model.h5', custom_objects={'vgg_loss': vgg_loss.vgg_loss})
        discriminator = load_model('./srgan_model/' + load + '_dis_model.h5')
        print("loaded gan model >> " + load + "_gen_model.h5")
        print("loaded dis model >> " + load + "_dis_model.h5")
        # loss_file = open('./srgan_model/losses.txt' , 'w+')
        # loss_file.close()

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # Adam
    generator.compile(loss=vgg_loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    
    # generator.summary()
    # discriminator.summary()
    # time.sleep(200)
    
    gan = get_gan_network(discriminator, lr_shape, generator, optimizer, vgg_loss.vgg_loss)
    print('\n', ':'*50)
    print('', ':'*16, "  model defined  ", ':'*15)
    print('', ':'*50, '\n')

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            
            # rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            # image_batch_hr = x_train_hr[rand_nums]
            # image_batch_lr = x_train_lr[rand_nums]
            fall_rand_nums = np.random.randint(0, len(fall_lr), size=int(batch_size/2))
            nonfall_rand_nums = np.random.randint(0, len(nonfall_lr), size=int(batch_size/2))
            image_batch_lr, image_batch_hr = get_batch_data(fall_lr, fall_lr_label, fall_hr_dict, fall_rand_nums,
                                                            nonfall_lr, nonfall_lr_label, nonfall_hr_dict, nonfall_rand_nums)
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
            # rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            # image_batch_hr = x_train_hr[rand_nums]
            # image_batch_lr = x_train_lr[rand_nums]
            fall_rand_nums = np.random.randint(0, len(fall_lr), size=int(batch_size/2))
            nonfall_rand_nums = np.random.randint(0, len(nonfall_lr), size=int(batch_size/2))
            image_batch_lr, image_batch_hr = get_batch_data(fall_lr, fall_lr_label, fall_hr_dict, fall_rand_nums,
                                                            nonfall_lr, nonfall_lr_label, nonfall_hr_dict, nonfall_rand_nums)

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            
        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)
        
        loss_file = open('./srgan_loss/' + save + 'losses.txt' , 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, discriminator_loss) )
        loss_file.close()

        # if e == 1 or e % 5 == 0:
        #     Utils.plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr)
        if e % 20 == 0:
            generator.save('./srgan_model/' + save + '_e%d_gen_model.h5' % e)
            discriminator.save('./srgan_model/' + save + '_e%d_dis_model.h5' % e)


if __name__== "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=64, help='Batch Size', type=int)
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=9999, help='number of iteratios for trainig', type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--load', help='model weight name to load', default=None, type=str)
    parser.add_argument('--save', help='model weight name to save', default='best', type=str)
    args = parser.parse_args()
    
    train(args.batch_size, args.epochs, args.lr, args.load, args.save)

