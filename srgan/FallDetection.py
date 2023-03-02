import numpy as np
import os
from scipy import io
import random
import argparse

import tensorflow as tf
from keras import layers, models, optimizers, losses
from keras.applications.vgg19 import VGG19
from keras.models import Model, load_model

from SrganDataLoader import *
from Network import *
from Parameters import *
from VggLoss import VGGLOSS
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
        
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class FallDetection:
    def __init__(self, learning_rate: float, epochs: int) -> None:
        self.adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.bce_loss = losses.BinaryCrossentropy(from_logits=False)
        self.epochs = epochs
        pass

    def fall_detect_model(self, csi_shape): # transfer learning with VGG19 model
        model = models.Sequential()
        vgg19 = VGG19(input_shape=csi_shape, weights='imagenet', include_top=False) #Training with Imagenet weights
        vgg19_layer_list = vgg19.layers
        
        for i in range(len(vgg19_layer_list)-1):
            model.add(vgg19_layer_list[i])
        for vgg_layer in model.layers:
            vgg_layer.trainable = False
            
        model.add(layers.BatchNormalization(momentum=0.5))
        model.add(layers.Flatten())
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(units=1024, activation='leaky_relu'))
        model.add(layers.Dense(units=64, activation='leaky_relu'))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        return model

    def make_mat_to_image(self, path, file):
        nonfall_te_matfile = io.loadmat(os.path.join(path,file))
        result = nonfall_te_matfile["result"] # load csi data from .mat file
        csidata = np.array(result)
        resize_shape = (Parameters.hr_image_shape[1], Parameters.hr_image_shape[0])
        if csidata.shape[0] == (Parameters.lr_sample_rate*Parameters.activity_duration):
            resize_shape = (Parameters.lr_image_shape[1], Parameters.lr_image_shape[0]) # image is (H x W x C), dsize=(W, H)        

        image = cv2.resize(csidata, dsize=resize_shape, interpolation=cv2.INTER_CUBIC)
        return image
    


    def get_training_data(self):
        fall_tr_path = '../dataset/fall_train_wden_pca'
        nonfall_tr_path = '../dataset/nonfall_train_wden_pca'
        fall_tr_files = os.listdir(fall_tr_path)
        nonfall_tr_files = os.listdir(nonfall_tr_path)
        x, y = self.return_data(fall_tr_path, nonfall_tr_path, fall_tr_files, nonfall_tr_files)
        
        tp = list(zip(x, y))
        random.shuffle(tp)
        train_x, train_y = zip(*tp)
            
        return np.array(train_x), np.array(train_y)

    def get_test_data(self):
        fall_te_path = '../dataset/fall_test_wden_pca'
        nonfall_te_path = '../dataset/nonfall_test_wden_pca'
        fall_te_files = os.listdir(fall_te_path)
        nonfall_te_files = os.listdir(nonfall_te_path)
        
        x, y = self.return_data(fall_te_path, nonfall_te_path, fall_te_files, nonfall_te_files)
        return x, y
    
    def get_ds_training_data(self, ds_rate):
        fall_tr_path = '../dataset/ds_train_data/ds_fall_train_' + str(ds_rate) + '_wden_pca'
        nonfall_tr_path = '../dataset/ds_train_data/ds_nonfall_train_' + str(ds_rate) + '_wden_pca'
        fall_tr_files = os.listdir(fall_tr_path)
        nonfall_tr_files = os.listdir(nonfall_tr_path)
        x, y = self.return_data(fall_tr_path, nonfall_tr_path, fall_tr_files, nonfall_tr_files)
        
        tp = list(zip(x, y))
        random.shuffle(tp)
        train_x, train_y = zip(*tp)
            
        return np.array(train_x), np.array(train_y)

    def get_ds_test_data(self, ds_rate):
        fall_te_path = '../dataset/ds_test_data/ds_fall_test_' + str(ds_rate) + '_wden_pca'
        nonfall_te_path = '../dataset/ds_test_data/ds_nonfall_test_' + str(ds_rate) + '_wden_pca'
        fall_te_files = os.listdir(fall_te_path)
        nonfall_te_files = os.listdir(nonfall_te_path)
        
        x, y = self.return_data(fall_te_path, nonfall_te_path, fall_te_files, nonfall_te_files)
        return x, y
    
    def return_data(self, fall_path, nonfall_path, fall_files, nonfall_files):
        x, y = [], []
        for filename in fall_files:
            image = self.make_mat_to_image(fall_path, filename)
            x.append(image)
            y.append(1) # fall == 1, nonfall = 0
            
        for filename in nonfall_files:
            image = self.make_mat_to_image(nonfall_path, filename)
            x.append(image)
            y.append(0) # fall == 1, nonfall = 0
            
        return np.array(x), np.array(y)


    def train_cnn(self, load: str, save: str):
        x_train, y_train = self.get_training_data()
        x_test, y_test = self.get_test_data()
        
        model = self.fall_detect_model(csi_shape = Parameters.hr_image_shape)
        model.compile(loss=self.bce_loss, optimizer=self.adam_opt, metrics=['accuracy'])
        
        if load is not None:
            model.load_weights('./fall_detect_model/' + load + "_weights")
        
        if self.epochs != 0:
            history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=64)
            model.save_weights("./fall_detect_model/" + save + "_weights")
        
        test_results = model.evaluate(x_test, y_test, verbose=1)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')
        
        result = np.zeros(shape=(2, 2), dtype=int)
        print(x_test.shape)
        for i, x in enumerate(x_test):
            pred = model.predict(np.expand_dims(x, axis=0))
            print(pred, y_test[i])
            if pred >= 0.5:
                pred = 1
            else:
                pred = 0
            result[y_test[i], pred] += 1
        print(result)
        
    def train_ds_cnn(self, load: str, save: str):
        x_train, y_train = self.get_ds_training_data(Parameters.lr_sample_rate)
        x_test, y_test = self.get_ds_test_data(Parameters.lr_sample_rate)
        
        model = self.fall_detect_model(csi_shape = Parameters.lr_image_shape)
        model.compile(loss=self.bce_loss, optimizer=self.adam_opt, metrics=['accuracy'])
        
        if load is not None:
            model.load_weights('./ds_fall_detect_model/' + load + "_" + str(Parameters.lr_sample_rate) + "_weights")
        
        if self.epochs != 0:
            history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=64)
            model.save_weights("./ds_fall_detect_model/" + save + "_" + str(Parameters.lr_sample_rate) + "_weights")
        
        test_results = model.evaluate(x_test, y_test, verbose=1)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')
        
        result = np.zeros(shape=(2, 2), dtype=int)
        for i, x in enumerate(x_test):
            pred = model.predict(np.expand_dims(x, axis=0))
            print(pred, y_test[i])
            if pred >= 0.5:
                pred = 1
            else:
                pred = 0
            result[y_test[i], pred] += 1
        print(result)
        
        
    def test_from_srgan(self):
        x_test, y_test = self.get_ds_test_data(Parameters.lr_sample_rate)
        # gen = Generator()
        vgg_loss = VGGLOSS(Parameters.hr_image_shape)
        generator = load_model('./srgan_model/test_50hz_e100_gen_model.h5', custom_objects={'vgg_loss': vgg_loss.vgg_loss})
        x_sr = generator.predict(x_test)
        print(x_test.shape)
        print(x_sr.shape)
        
        cnnmodel = self.fall_detect_model(csi_shape = Parameters.hr_image_shape)
        cnnmodel.compile(loss=self.bce_loss, optimizer=self.adam_opt, metrics=['accuracy'])
        
        cnnmodel.load_weights("./fall_detect_model/e200_weights")
        
        test_results = cnnmodel.evaluate(x_sr, y_test, verbose=1)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')
        
        result = np.zeros(shape=(2, 2), dtype=int)
        for i, x in enumerate(x_sr):
            pred = cnnmodel.predict(np.expand_dims(x, axis=0))
            print(pred, y_test[i])
            if pred >= 0.5:
                pred = 1
            else:
                pred = 0
            result[y_test[i], pred] += 1
        print(result)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--epochs", type=int, default=0)
    parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--load', help='model weight name to load', default=None, type=str)
    parser.add_argument('--save', help='model weight name to save', default='best', type=str)
    args = parser.parse_args()
    
    fall_detect = FallDetection(args.lr, args.epochs)
    # fall_detect.train_cnn(args.load, args.save)
    # fall_detect.train_ds_cnn(args.load, args.save)
    fall_detect.get_training_pair()
    
    # model = define_cnn([500, 10, 3])
    # model.summary()
            
if __name__=='__main__':
    main()