#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:04:49 2023

@author: ml_ai
"""

import utils
import os
import rasterio as rs
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil
from glob import glob
import numpy as np
import sys
# import constants as C
# import streamlit as st
# in_dir = C.origImageDir
# aug_dir = r'./work/augData'

# images_dir = r'./work/augData/images'
# masks_dir = r'./work/augData/masks'

# dir_struct_dict = {'train_images': r'./work/splitData/train_images',
#               'test_images' : r'./work/splitData/test_images',
#               'val_images' : r'./work/splitData/val_images',
#               'train_masks': r'./work/splitData/train_masks',
#               'test_masks': r'./work/splitData/test_masks',
#               'val_masks': r'./work/splitData/val_masks'}
# user_seed=11
# user_patch_size = 512

    
def genAugData(in_dir, images_dir, masks_dir, user_patch_size, visual=False, padding=False):    
    utils.createDir(images_dir)
    utils.createDir(masks_dir)
    
    image_mask_dict=defaultdict()
    imageList = [f for f in os.listdir(in_dir) if f.endswith('_image.tif')] #All TIFF Images

    for f in imageList:
        k = f.split('_image')[0]
        if os.path.exists(os.path.join(in_dir, '{}_mask.tif'.format(k))):
            image_mask_dict[os.path.join(in_dir, f)] = os.path.join(in_dir, '{}_mask.tif'.format(k)) #Dict of {Image:Mask, ...}
    
    count=0
    
    image_scaler,mask_scaler = utils.getScaler(image_mask_dict)
    
    for inImgFile,inMaskFile in image_mask_dict.items():
        count+=1
        image_dataset, mask_dataset = utils.patchImages(image_scaler, mask_scaler, inImgFile, inMaskFile, patch_size=user_patch_size, padding=padding)
         
        for i in range(len(image_dataset)):
        
            out_image = os.path.join(images_dir, "{}_aug_{}.tif".format(count, i))
            utils.array2Tif(image_dataset[i], out_image, proj=None, transform=None, nodatavalue=None)
            
            out_mask = os.path.join(masks_dir,"{}_aug_{}.tif".format(count, i))
            utils.array2Tif(mask_dataset[i], out_mask, proj=None, transform=None, nodatavalue=None)
                
            if visual:
                utils.visualize_image_mask(image_dataset[i], mask_dataset[i])

def createSplitDir(dir_struct_dict):
    for k,v in dir_struct_dict.items():
        utils.createDir(os.path.dirname(v))
        utils.createDir(v)
        if k.startswith('train'):
            utils.createDir(os.path.join(v, 'train'))
        elif k.startswith('test'):
            utils.createDir(os.path.join(v, 'test'))
        elif k.startswith('val'):
            utils.createDir(os.path.join(v, 'val'))
            
def createSplitDirVR(dir_struct_dict):
    for k,v in dir_struct_dict.items():
        utils.createDir(os.path.dirname(v))
        utils.createDir(v)        
        if k.startswith('train'):
            utils.createDir(os.path.join(v, 'train'))
        elif k.startswith('val'):
            utils.createDir(os.path.join(v, 'val'))
            
def index2dir(filenames, indir, masks_dir, outdir, Isimages):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i in filenames:
            if Isimages==True:
                if not os.path.exists(os.path.join(os.path.join(outdir, '{}'.format(i)))):
                    shutil.move(os.path.join(indir, '{}'.format(i)),  outdir)
            else:
                if not os.path.exists(os.path.join(os.path.join(outdir, '{}'.format(i)))):
                    shutil.move(os.path.join(masks_dir, '{}'.format(i)), outdir)

def splitData(images_dir, masks_dir, dir_struct_dict, user_seed, ttv=None, validation_ratio=None):
    if validation_ratio:        
        createSplitDirVR(dir_struct_dict)
    #ttv: Train , test and validation ratio respectively
    
    else:
        createSplitDir(dir_struct_dict)
        
    #Splitting into train, test and validation
    filenames = glob(os.path.join(images_dir, '*aug*.tif'))
    filenames = np.array([os.path.basename(i) for i in filenames])
    nfiles = len(filenames)
    indexes = np.arange(nfiles)
    
    if validation_ratio:
        rng=np.random.default_rng(seed=user_seed)
        rng.shuffle(indexes)
        train_index = indexes[:int(validation_ratio*nfiles)]
        val_index = indexes[int(validation_ratio*nfiles):]
    
        #store splitted images into separate directory to give in ImageGenerator
        index2dir(filenames[train_index], images_dir, masks_dir, os.path.join(dir_struct_dict['train_images'],'train'), True)
        index2dir(filenames[train_index], masks_dir,  masks_dir, os.path.join(dir_struct_dict['train_masks'],'train'), False)
        
        index2dir(filenames[val_index], images_dir,  masks_dir, os.path.join(dir_struct_dict['val_images'],'val'), True)
        index2dir(filenames[val_index], masks_dir,  masks_dir, os.path.join(dir_struct_dict['val_masks'],'val'), False)

    else:
               
        # Rndom Shuffling ###
        rng=np.random.default_rng(seed=user_seed)
        rng.shuffle(indexes)
        train_index = indexes[:int(ttv[0]*nfiles)]
        val_index= indexes[int(ttv[0]*nfiles):int((ttv[0] + ttv[1])*nfiles)]
        test_index = indexes[int((ttv[0] + ttv[1])*nfiles):]
        # print(train_index, test_index)
        
        #store splitted images into separate directory to give in ImageGenerator
        index2dir(filenames[train_index], images_dir, os.path.join(dir_struct_dict['train_images'],'train'), True)
        index2dir(filenames[train_index], masks_dir, os.path.join(dir_struct_dict['train_masks'],'train'), False)
        
        index2dir(filenames[test_index], images_dir, os.path.join(dir_struct_dict['test_images'],'test'), True)
        index2dir(filenames[test_index], masks_dir,  os.path.join(dir_struct_dict['test_masks'],'test'), False)
        
        index2dir(filenames[val_index], images_dir, os.path.join(dir_struct_dict['val_images'],'val'), True)
        index2dir(filenames[val_index], masks_dir, os.path.join(dir_struct_dict['val_masks'],'val'), False)
        

    

def prepare_data(ST_MSG, trainingDir, images_dir, masks_dir, dir_struct_dict, user_seed, user_patch_size, ttv=None, validation_ratio=0.8, aug=False,split=False, padding=False):
    utils.createDir(os.path.dirname(images_dir))
    
    if aug:
        ST_MSG.markdown("**:green[Data Preparation]**, :blue[Augmenting Data ...]")
        genAugData(trainingDir, images_dir, masks_dir, user_patch_size, visual=False, padding=False)
    if split:
        ST_MSG.markdown("**:green[Data Preparation]**, :blue[Splitting Data ...]")
        splitData(images_dir, masks_dir, dir_struct_dict, user_seed, ttv=None, validation_ratio=0.8)
        
        
