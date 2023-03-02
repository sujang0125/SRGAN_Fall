from scipy import io
import numpy as np
import tensorflow as tf
import os
import cv2
import math
import time
import Parameters
import warnings
import pickle

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class SrganDataLoader:
    def __init__(self) -> None:
        self.hr_rate = Parameters.hr_sample_rate # high resolution rate=1000Hz
        self.lr_rate = Parameters.lr_sample_rate # down-sampling rates

    def save_srgan_train_data(self) -> None:
        hr_fall_tr_path = "../dataset/fall_train_wden_pca/"
        hr_nonfall_tr_path = "../dataset/nonfall_train_wden_pca/"
        lr_fall_tr_path = "../dataset/augmented/fall_tr_aug_" + str(self.lr_rate) + "_wden_pca/"
        lr_nonfall_tr_path = "../dataset/augmented/nonfall_tr_aug_" + str(self.lr_rate) + "_wden_pca/"

        fall_lr, fall_lr_label, fall_hr_dict = self.get_train_data(hr_fall_tr_path, lr_fall_tr_path)
        print("fall data loaded, len ", len(fall_lr), len(fall_lr_label), len(fall_hr_dict)) 
        print("fall data loaded, shape ", fall_lr[0].shape, fall_hr_dict[0].shape)
        nonfall_lr, nonfall_lr_label, nonfall_hr_dict = self.get_train_data(hr_nonfall_tr_path, lr_nonfall_tr_path)
        print("nonfall data loaded, len ", len(nonfall_lr), len(nonfall_lr_label), len(nonfall_hr_dict)) 
        print("nonfall data loaded, shape ", nonfall_lr[0].shape, nonfall_hr_dict[0].shape)
        
        np.save('./training_data/' + str(Parameters.lr_sample_rate) + '_fall_lr.npy', fall_lr)
        np.save('./training_data/' + str(Parameters.lr_sample_rate) + '_fall_lr_label.npy', fall_lr_label)
        np.save('./training_data/' + str(Parameters.lr_sample_rate) + '_nonfall_lr.npy', nonfall_lr)
        np.save('./training_data/' + str(Parameters.lr_sample_rate) + '_nonfall_lr_label.npy', nonfall_lr_label)
        with open('./training_data/' + str(Parameters.lr_sample_rate) + '_fall_hr_dict.pickle','wb') as fw:
            pickle.dump(fall_hr_dict, fw)
        with open('./training_data/' + str(Parameters.lr_sample_rate) + '_nonfall_hr_dict.pickle','wb') as fw:
            pickle.dump(nonfall_hr_dict, fw)
        

    def matfile_to_image(self, matfile_path: str, matfile_name: str) -> np.ndarray:
        """
            get .mat file and normalize to RGB image value, return with numpy ndarray
            Args:
                matfile_path (str): directory (path) of .mat file
                matfile_name (str): name of .mat file
            Returns:
                numpy array of normalized image
        """
        nonfall_te_matfile = io.loadmat(matfile_path + matfile_name)
        result = nonfall_te_matfile["result"] # load csi data from .mat file
        csidata = np.array(result)
        
        resize_shape = (Parameters.hr_image_shape[1], Parameters.hr_image_shape[0])
        if csidata.shape[0] == (self.lr_rate*Parameters.activity_duration):
            resize_shape = (Parameters.lr_image_shape[1], Parameters.lr_image_shape[0])

        image = cv2.resize(csidata, dsize=resize_shape, interpolation=cv2.INTER_CUBIC)
            
        return image

    def get_train_data(self, hr_path: str, lr_path: str):
        """
            return training data list of [lowrate_image, target_image]
            Args:
                hr_path (str): target file's path
                lr_path (str): lowrate file's path
            Returns:
                result (list): list of a pair of [low sampled image, target image]
        """
        lr_images_list, lr_label = [], []
        hr_images_dict = dict()
        hr_list = os.listdir(hr_path) # list of file in target path
        for i, hr_filename in enumerate(hr_list):
            if i >= 10: 
                break
            hr_image = self.matfile_to_image(hr_path, hr_filename)
            hr_images_dict[i] = hr_image
            lr_list = os.listdir(lr_path) # list of file in low-rate sampled path
            for lr_filename in lr_list:
                if lr_filename.startswith(hr_filename[:-12]): # find the same csi data name, it is the pair
                    lr_image = self.matfile_to_image(lr_path, lr_filename)
                    # result.append([lr_image, hr_image]) 
                    lr_images_list.append(lr_image)
                    lr_label.append(i)
            if i % 50 == 0:
                print("train data loaded length == ", len(lr_images_list))
                
        return lr_images_list, lr_label, hr_images_dict

    # fall_tr_matfilelist = [file for file in fall_tr_augmented if file.startswith(hr_filename[:-13])]
    # nonfall_tr_matfilelist = [file for file in nonfall_tr_augmented if file.endswith(".mat")]       
    # print ("file_list_py: {}".format(fall_tr_mat))

if __name__=="__main__":
    dataloader = SrganDataLoader()
    dataloader.save_srgan_train_data()

    
    
# import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

# class CustomTrainDataset(Dataset): 
#     def __init__(self):
#         self.srgan_data_loader = SrganDataLoader(lr_rate=100, activity_duration=10)
#         srgan_train_data = self.srgan_data_loader.get_srgan_train_data()
#         self.train_data = srgan_train_data # np.ndarray with shape of (num_train_data x 2)
#         self.num_train = srgan_train_data.shape[0]

#     # 총 데이터의 개수를 리턴
#     def __len__(self): 
#         return self.num_train

#     # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
#     def __getitem__(self, idx): 
#         x = torch.from_numpy(self.train_data[idx][0]) # low sampled data e.g. lr_rate
#         y = torch.from_numpy(self.train_data[idx][1]) # target data e.g. 1000 Hz
#         return x, y

# class CustomTestDataset(Dataset): 
#     def __init__(self):
#         self.srgan_data_loader = SrganDataLoader(lr_rate=100, activity_duration=10)
#         srgan_test_data = self.srgan_data_loader.get_srgan_test_data()
#         self.test_data = srgan_test_data # np.ndarray with shape of (num_test_data x 2)
#         self.num_test = srgan_test_data.shape[0]

#     # 총 데이터의 개수를 리턴
#     def __len__(self): 
#         return self.num_test

#     # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
#     def __getitem__(self, idx): 
#         x = torch.from_numpy(self.test_data[idx][0]) # low sampled data e.g. lr_rate
#         y = torch.from_numpy(self.test_data[idx][1]) # target data e.g. 1000 Hz
#         return x, y