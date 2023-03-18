#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 12:01:48 2023

@author: ml_ai
"""

import utils
import os
from glob import glob
from tensorflow.keras.models import load_model
import numpy as np
import tifffile
from patchify import patchify,unpatchify
import DataGenerator
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import shutil
import pickle as pkl

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def genTestData(inImgFile, images_dir, user_patch_size, visual=True):    
    utils.createDir(images_dir)
    
    with open(r'./image_scaler.pkl', 'rb') as f:
        image_scaler = pkl.load(f)
    with open(r'./mask_scaler.pkl', 'rb') as f:
        mask_scaler = pkl.load(f)
    
    
    image_dataset  = utils.patchImages(image_scaler, mask_scaler, inImgFile, patch_size=user_patch_size, padding=True, augment=False)
         
    for i in range(len(image_dataset)):
    
        out_image = os.path.join(images_dir, "aug_{}.tif".format(i))
        utils.array2Tif(image_dataset[i], out_image, proj=None, transform=None, nodatavalue=None)

        if visual:
            utils.visualize_image(image_dataset[i])
    
    return image_dataset
            
#genTestData()

def predictTestData(modelpath, images_dir, batch_size=1):
    print(images_dir)

    # testGenerator = Gen(glob(os.path.join(images_dir, '*.tif')), 
    #                      glob(os.path.join(images_dir, '*.tif')), batch_size=user_batch_size)
    
    img_files = glob(os.path.join(images_dir, '*.tif'))
    img_files.sort(key=lambda f: int(os.path.basename(f)[:-4].split('_')[1]))
    
    array_files = [tifffile.imread(i) for i in img_files]
    x = np.array(array_files)
    
    print(x.shape)
    
    # testGenerator = DataGenerator.Gen(img_files, img_files, batch_size=batch_size)
    # print(testGenerator.x[:10])
        
    model = load_model(modelpath, custom_objects={'jacard_coef_loss':jacard_coef_loss, 'jacard_coef':jacard_coef})
    # predictions_list = model.predict_generator(testGenerator)
    # print("88888888", predictions_list[0].shape)

    # for img in img_files:
    #     x = tifffile.imread(img)
    #     print(x.shape)
    #     y = model.predict(x)
    #     predictions_list.append(y)
        
    # predictions_list = [np.where(prediction>0.5, 255, 0).astype(np.uint8) for prediction in predictions_list]
    
    predictions = model.predict(x, batch_size=1)
    predictions = np.where(predictions>0.5, 1, 0)
    print(predictions.shape)
    
    predictions_list=[]
    for i in range(len(predictions)):
        try:
            # if (i==29):
            #     print(predictions[i,...])
            # print('Saving Mask',i)
            tifffile.imsave('./rs/mask_{}.tif'.format(i), predictions[i,...])
            predictions_list.append(predictions[i,...])
        except Exception as e:
            print(e)
        
        
        
    return predictions_list

#Combine all patched image
def makeprediction(inImgFile, maskFile, modelpath, user_batch_size, user_patch_size, patch=True):
    
    images_dirpath = os.path.join(os.path.dirname(inImgFile), 'testData')
    if os.path.isdir(images_dirpath):
        shutil.rmtree(images_dirpath)
    utils.createDir(images_dirpath)
       
    if patch:
        mask_dataset = genTestData(inImgFile, images_dirpath, user_patch_size, visual=False)
    predictions_list = predictTestData(modelpath, images_dirpath, batch_size=1)
    
    mask_path = utils.unpatchMasks(inImgFile, maskFile, predictions_list, patch_size=user_patch_size)



