#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:18:15 2023

@author: ml_ai
"""

import numpy as np
import os
import tifffile

import rasterio as rs # Used for reading writing raster files

from patchify import patchify, unpatchify# Used for splitting image
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import albumentations as A
import cv2

import pickle as pkl
import matplotlib.pyplot as plt

def getScaler(image_mask_dict, storeScaler=True):
    i=0
    for inImgFile,inMaskFile in image_mask_dict.items():
        x_image = tifffile.imread(inImgFile)
        y_mask = tifffile.imread(inMaskFile)
        
        x_image = x_image.reshape(-1, x_image.shape[-1])
        y_mask = y_mask.reshape(-1, y_mask.shape[-1])
        
        # print('>>>>>>>>>>>Image,Mask Shapes>>>>>',x_image.shape,y_mask.shape)
        
        if i==0:
            r_image = x_image
            r_mask = y_mask
        else:
            r_image = np.vstack((r_image,x_image))
            r_mask = np.vstack((y_mask,y_mask))
            
        i = i+1
        
    print('>>>>>>>>>>>>>>>>>>>>Shapes>>>>>',r_image.shape,r_mask.shape)
    scaler = MinMaxScaler()
    image_scaler, mask_scaler = scaler.fit(r_image), scaler.fit(r_mask)
    del r_image,r_mask
    
    if storeScaler:
        with open('./image_scaler.pkl', 'wb') as f:
            pkl.dump(image_scaler, f)
        with open('./mask_scaler.pkl', 'wb') as f:
            pkl.dump(mask_scaler, f)
            
    return image_scaler,mask_scaler
    

def array2Tif(array, tifFilePath, proj, transform, nodatavalue):
    
    with rs.open(tifFilePath, 'w', driver="GTIFF",
                            height= array.shape[0], width = array.shape[1],
                            count = array.shape[2], dtype=str(array.dtype),
                            crs = proj, transform = transform) as dst:
        
        for i in range(array.shape[2]):
            dst.write(array[:,:,i],i+1)

def createDir(dirName):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
        
def plotOGTPred(inImgFile, gtFile, maskFile, outFile):
    fig = plt.figure(figsize=(20,8))
    
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(tifffile.imread(inImgFile)[:,:,:4])
    ax1.title.set_text('Original Image')
    ax1.grid(False)
    
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('Ground Truth Mask')
    ax2.imshow(tifffile.imread(gtFile))
    ax2.grid(False)
    
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Predicted Mask')
    ax3.imshow(tifffile.imread(maskFile))
    ax3.grid(False)
    
    plt.savefig(outFile, facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 200)
    plt.show()

def visualize_image_mask(inImgFile, maskFile, outFile):
    fig = plt.figure(figsize=(20,8))
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(tifffile.imread(inImgFile)[:,:,:3])
    ax1.title.set_text('Original Image')
    ax1.grid(False)
    

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title('Predicted Mask')
    ax2.imshow(tifffile.imread(maskFile))
    ax2.grid(False)
    
    plt.savefig(outFile, facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 200)
    plt.show()
    
def visualize_act_pred(act, pred):
    plt.figure(figsize=(20, 10))
    plt.subplot(1,2,1)
    plt.imshow(act)
    plt.subplot(1,2,2)
    plt.imshow(pred)  
    
def visualize_image(image):
    image = image[:,:,:3]
    # print(image.shape)
    plt.figure(figsize=(20, 10))
    plt.imshow(image)
    

def augmentImage(image_array, mask_array, aug_count=4, patch_size=256):
    augmented_image_set = []
    augmented_mask_set = []
    

    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Rotate(limit=[60, 240], p=1.0, interpolation=cv2.INTER_NEAREST),
        #A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.4], contrast_limit=0.2, p=1.0),
        # A.OneOf([
        #     A.CLAHE (clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        #     A.GridDistortion(p=0.5),
        #     A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
        # ], p=1.0),splitData(images_dir, masks_dir, ttv=(0.8,0.1,0.1))
    ], p=1.0)
    
    print(image_array.shape, mask_array.shape) 
    
    for i in range(aug_count):
        
        transformed = transform(image=image_array, mask=mask_array)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        
        augmented_image_set.append(transformed_image)
        augmented_mask_set.append(transformed_mask)
        
        
    return augmented_image_set, augmented_mask_set

def addpadding(array, patch_size):
    pad_h = ((array.shape[0] // patch_size) + 1) * patch_size - array.shape[0]
    pad_w = ((array.shape[1] // patch_size) + 1) * patch_size - array.shape[1]
    
    #before_shape = array.shape
    print("before_padding", array.shape)
    array = np.pad(array,pad_width = ((0,pad_h),(0,pad_w),(0,0)))
    print("after_padding", array.shape)
    #after_shape = array.shape
    
    return array

# def removepadding(array, patch)


def patchImages(image_scaler, mask_scaler, image_path, mask_path=None, patch_size=256, padding=False, augment=True):
    
    patched_image_dataset = []        
    with rs.open(image_path, 'r') as src:
        arr_st = src.read()
        profile = src.profile        
    #print(arr_st.shape) #(C, H, W)
    image_features = np.moveaxis(arr_st, 0, -1)
    

    nImageBands = profile["count"]
    
    ## Add Padding
    if padding:
        image_features = addpadding(image_features, patch_size)
        
    if mask_path:
        patched_mask_dataset = []
        with rs.open(mask_path, 'r') as src:
            arr_st = src.read()
            profile = src.profile
            
        #print(arr_st.shape) #(C, H, W)
        mask_features = np.moveaxis(arr_st, 0, -1)
        nMaskBands = profile["count"]
        #print(features.shape) #(H, W, C)
        
        if padding:
            mask_features = addpadding(mask_features, patch_size)
    
    if augment:
        augmented_image_set, augmented_mask_set = augmentImage(image_features, mask_features)
        # print('>>>>>>>:',len(augmented_image_set))
    else:
        if mask_path:
            augmented_image_set, augmented_mask_set = [image_features], [mask_features]
        else:
            augmented_image_set = [image_features]
    
    # scaler = MinMaxScaler()
    # for i in augmented_image_set:
    #     scaler.fit(augmented_image_set[i].reshape(-1, augmented_image_set[i].shape[-1])) #(X*Y, 4)
        
        
    
    # print('>>>>>AImg',len(augmented_image_set),'AMask',len(augmented_mask_set))
    for n in range(len(augmented_image_set)):   
        # Scaling to augmented Image
        scaled_aug_image = image_scaler.fit_transform(augmented_image_set[n].reshape(-1, augmented_image_set[n].shape[-1])).reshape(augmented_image_set[n].shape)
        image_patches = patchify(scaled_aug_image, (patch_size, patch_size, nImageBands), step=patch_size)  #Step=256 for 256 patches means no overlap
    #         print(image_patches.shape) #(3, 5, 1, 256, 256, 4)
        if mask_path:
            scaled_aug_mask = mask_scaler.fit_transform(augmented_mask_set[n].reshape(-1, augmented_mask_set[n].shape[-1])).reshape(augmented_mask_set[n].shape)
            mask_patches = patchify(scaled_aug_mask, (patch_size, patch_size, nMaskBands), step=patch_size)  #Step=256 for 256 patches means no overlap
    #         print(image_patches.shape) #(3, 5, 1, 256, 256, 1)
        print(image_patches.shape)
    
        for i in range(image_patches.shape[0]):
            for j in range(image_patches.shape[1]):
                # print(i,j)
                single_patch_img = image_patches[i,j,0]
                #Use minmaxscaler instead of just dividing by 255. 
                #print(single_patch_img.shape) # (256, 256, 4)
                #print(single_patch_img.reshape(-1, single_patch_img.shape[-1]).shape) # (65536, 4)
                
                ##single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                patched_image_dataset.append(single_patch_img)
                
                if mask_path:               
                    single_patch_mask = mask_patches[i,j,0]
                    ##single_patch_mask = scaler.fit_transform(single_patch_mask.reshape(-1, single_patch_mask.shape[-1])).reshape(single_patch_mask.shape)
                    patched_mask_dataset.append(single_patch_mask)
                
    if mask_path:
        return patched_image_dataset, patched_mask_dataset
    else:
        return patched_image_dataset


def unpatchMasks(image_path, mask_path, mask_dataset, patch_size=256):
    
    predicted_patches_arr = np.array(mask_dataset)       
    with rs.open(image_path, 'r') as src:
        img_h = src.height
        img_w = src.width
        
    print(predicted_patches_arr.shape)
               
    # image_features = np.moveaxis(arr_st, 0, -1)
    
    # img_h = image_features.shape[0]
    # img_w = image_features.shape[1]
    

    out_h = ((img_h// patch_size)+1)*patch_size
    out_w = ((img_w // patch_size)+1)*patch_size

    n_h = int(out_h/patch_size)
    n_w = int(out_w/patch_size)
    
    print(n_h, n_w, patch_size, patch_size)
    
    predicted_patches_reshaped = np.reshape(predicted_patches_arr,(n_h, n_w, patch_size, patch_size))
    reconstructed_image = unpatchify(predicted_patches_reshaped,(out_h,out_w))
    print('re', reconstructed_image.shape)
    
    ##########Reconstructing original array size same as input###################

    final_arr_output = np.zeros((img_h ,img_w))

    for i in range(img_h):
        for j in range(img_w):
            final_arr_output[i,j] = reconstructed_image[i,j]
    print(final_arr_output.shape)
   
    #############################################################################
    
    mask_meta = src.meta.copy()
    
    mask_meta.update({'driver': 'GTiff', 'count':1, 'dtype':'uint8'})
    with rs.open(mask_path, 'w', **mask_meta) as dst:
        dst.write(final_arr_output, 1)
        
    return mask_path
    
    

    

