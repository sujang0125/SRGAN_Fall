import cv2
import numpy as np
from scipy import io
import os
import math
import Parameters
import random


# random.seed(42)
# a = np.array([-1])
# for i in range(100):
#     rand_nums = np.random.sample(range(0,1000),10) 
#     for num in rand_nums:
#         if num in a:
#             print("er", num)
#     a = np.concatenate([a, rand_nums], axis=0)
#     # print(rand_nums)

# img = cv2.imread('/home/jsw/Downloads/toifall_dataset/empty/empty_023.jpg')
# arr = np.array(img)
# print(arr.shape)
# # 180 x 500 x 3

# st = "abcdefghijklmnop"
# a = st[:-10]
# print(a)

# arr = np.array([[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],[2, 5],[2, 5],[2, 5],[2, 5],[2, 5]])
# print(arr.shape)
# x = arr[:, 0]
# y = arr[:, 1]
# print(x)
# print(y)

# image = np.zeros(shape=(resize_shape[0], resize_shape[1], 3))
# # normalize data to a range between (0 - 1)
# for i in range(3):
#     ant = csidata[:,:,i]
#     ant_nor = (ant - np.min(ant)) / (np.max(ant) - np.min(ant))
#     ant_resize = cv2.resize(ant_nor, dsize=resize_shape, interpolation=cv2.INTER_CUBIC)
#     image[:,:,i] = ant_resize

# cv2.imwrite('color_img.jpg', csidata_nor)
# cv2.imshow("image", csidata_nor)
# cv2.waitKey(5000)

# a = np.zeros(shape=(10000, 10), dtype=np.int32
# while a.shape[1] <= a.shape[0]:
#     if a.shape[0] % 2 == 1:
#         break
#     a_b = a[:(a.shape[0]//2),:]
#     a_a = a[(a.shape[0]//2):,:]
#     a = np.append(a_b, a_a, axis=1) #np.concatenate([a_b, a_a])
#     print(a_b.shape)
#     print(a_a.shape)
#     print(a.shape)
# print(a.shape)

# a = dict()
# da = np.zeros(shape=(2, 3))
# a[0] = da
# a[1] = "def"
# print(a[0].shape)

# from tensorflow.python.client import device_lib
# import tensorflow as tf
# print(device_lib.list_local_devices())
# print()
# print()  
# print(tf.test.gpu_device_name())
# print()
# print()

from keras.applications.vgg19 import VGG19
from keras.models import Model, load_model
from VggLoss import VGGLOSS
from skimage.metrics import structural_similarity as ssim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def make_mat_to_image(path, file):
    nonfall_te_matfile = io.loadmat(os.path.join(path,file))
    result = nonfall_te_matfile["result"] # load csi data from .mat file
    csidata = np.array(result)
    resize_shape = (Parameters.hr_image_shape[1], Parameters.hr_image_shape[0])
    

    if csidata.shape[0] == (Parameters.lr_sample_rate*Parameters.activity_duration):
        resize_shape = (Parameters.lr_image_shape[1], Parameters.lr_image_shape[0]) # image is (H x W x C), dsize=(W, H)        


    image = cv2.resize(csidata, dsize=resize_shape, interpolation=cv2.INTER_CUBIC)
    return image

def normalize_rgb(data):    
    nor = np.ones(shape=data.shape)
    for i in range(3):
        nor[:,:,i] = (data[:,:,i] - np.min(data[:,:,i]))/(np.max(data[:,:,i])-np.min(data[:,:,i]))*255
    return nor

vgg_loss = VGGLOSS(Parameters.hr_image_shape)
generator = load_model('./srgan_model/test_50hz_e100_gen_model.h5', custom_objects={'vgg_loss': vgg_loss.vgg_loss})
# x_sr = generator.predict(x_test)

fall_tr_path = '../dataset/fall_train_wden_pca/'
fall_tr_files = os.listdir(fall_tr_path)
fall_ds_tr_path = '../dataset/ds_train_data/ds_fall_train_' + str(Parameters.lr_sample_rate) + '_wden_pca/'
fall_ds_tr_files = os.listdir(fall_ds_tr_path)
for file in fall_tr_files:
    image = make_mat_to_image(fall_tr_path, file)
    for dsfile in fall_ds_tr_files:
        if dsfile.startswith(file[:-12]):
            dsimage = make_mat_to_image(fall_ds_tr_path, dsfile)
            srimage = generator.predict(np.expand_dims(dsimage, axis=0))
            srimage = np.squeeze(srimage, axis=0)
            print(srimage.shape)
            break
    
    lrnor = normalize_rgb(dsimage)
    hrnor = normalize_rgb(image)
    srnor = normalize_rgb(srimage)
    # ssim_1, diff1 = ssim(hrnor, lrnor, channel_axis=2, full=True)
    # diff1 = (diff1 * 255).astype("uint8")
    # plt.imshow(diff1)
    ssim_2, diff2 = ssim(hrnor, srnor, channel_axis=2, full=True,multichannel=True)
    diff2 = (diff2 * 255).astype("uint8")

    print(ssim_2)
    # 0.21075336301148573 0.6888119020545118 0.7808179172891382
    
    # cv2.imshow("hr image", hrnor)
    # cv2.imshow("lr image", lrnor)
    # cv2.imshow("sr image", srnor)
    # cv2.imwrite(filename=file + "_lr_image.jpg", img=lrnor)
    # cv2.imwrite(filename=file + "_hr_image.jpg", img=hrnor)
    # cv2.imwrite(filename=file + "_sr_image.jpg", img=srnor)
    # cv2.waitKey(5000)