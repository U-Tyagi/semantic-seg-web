#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 18:09:40 2023

@author: ml_ai
"""
import os
from glob import glob
import utils
import shutil
import numpy as np
import pickle

### Importing DL libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import applications, optimizers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler, TensorBoard
#from tensorflow.keras.layers import Input, Lambda, Activation, Concatenate, Conv2D, MaxPooling2D, BatchNormalization, Add, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.layers import Input, Lambda, Activation, Concatenate, Conv2D, MaxPooling2D, BatchNormalization, concatenate, Conv2DTranspose, Dropout, add, multiply, UpSampling2D
from tensorflow.keras.applications import VGG16

from customLog import TrainingPlot

import DataGenerator

# input_shape = (512, 512, 4)
# user_seed=11
# user_batch_size=4
# numEpoch=20

# n_filters=32
# kernel_size=3
# dropout=0.1
# batch_norm=True
# depth=4

# dir_struct_dict = {'train_images': r'./work/splitData/train_images',
#               'test_images' : r'./work/splitData/test_images',
#               'val_images' : r'./work/splitData/val_images',
#               'train_masks': r'./work/splitData/train_masks',
#               'test_masks': r'./work/splitData/test_masks',
#               'val_masks': r'./work/splitData/val_masks'}


#%%

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

def getCallbacks(numEpochs, proj_absPath, archtype, STPLOT, STMSG):
    projname = os.path.basename(proj_absPath)
    exponential_decay_fn = exponential_decay(0.0001, numEpochs)
    
    lr_scheduler = LearningRateScheduler(
        exponential_decay_fn, #function
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        filepath = os.path.join(proj_absPath, '{}.h5'.format(projname)),
        save_best_only = True,
    #     save_weights_only = False,
        monitor = 'val_loss',
        mode = 'auto',
        verbose = 1
    )
    
    earlystop = EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.001,
        patience = 6,
        mode = 'auto',
        verbose = 1,
        restore_best_weights = True
    )
    
    
    csvlogger = CSVLogger(
        filename= os.path.join(proj_absPath, "{}_training_csv.log".format(archtype)),
        separator = ",",
        append = False
    )
    
    callbacks = [checkpoint, earlystop, csvlogger, lr_scheduler, TrainingPlot(projAbsPath = proj_absPath, stplot = STPLOT, stmsg = STMSG)]
    
    return callbacks

#%%
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def attention2D(inputs, skip_connection):
    theta_x = Conv2D(filters=1, kernel_size=1, activation='relu', padding='same')(inputs)
    phi_skip = Conv2D(filters=1, kernel_size=1, activation='relu', padding='same')(skip_connection)
    
    add_phix_theta_x = add([theta_x, phi_skip])
    
    f = Activation('sigmoid')(add_phix_theta_x)
    rate= Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding='same')(f)
    
    att_x = multiply([rate, skip_connection])
    return att_x


def unetAttn(input_shape, n_filters=32, kernel_size=3, dropout=0.1, batch_norm=True, depth=4):
    skips = []
    input_layer = Input(shape=input_shape)
    x = input_layer
    # Encoder
    for i in range(depth):
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        skips.append(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(rate=dropout)(x)
        n_filters = n_filters*2
    
    # Bottleneck
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
    if batch_norm:
        x = x = BatchNormalization()(x)
    x = Dropout(rate=dropout)(x)
    
    # Decoder
    for i in reversed(range(depth)):
        n_filters = n_filters//2
        x = Conv2DTranspose(filters=n_filters, kernel_size=3, strides = 2, padding = 'same')(x)
        #x = UpSampling2D(size=(2,2)(x)
        att_x = attention2D(x, skips[i])
        x = concatenate([x, att_x])
        
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        if batch_norm:
            x = x = BatchNormalization()(x)
        x = Dropout(rate=dropout)(x)

    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss = [jacard_coef_loss], metrics=[jacard_coef]) #binary_crossentropy
    model.summary()
    return model
    
    

def unet(input_shape, n_filters=32, kernel_size=3, dropout=0.1, batch_norm=True, depth=4):
    skips = []
    input_layer = Input(shape=input_shape)
    x = input_layer
    # Encoder
    for i in range(depth):
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        skips.append(x)
        if batch_norm:
            x = x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(rate=dropout)(x)
        n_filters = n_filters*2
    
    # Bottleneck
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
    if batch_norm:
        x = x = BatchNormalization()(x)
    x = Dropout(rate=dropout)(x)
    
    # Decoder
    for i in reversed(range(depth)):
        n_filters = n_filters//2
        x = Conv2DTranspose(filters=n_filters, kernel_size=3, strides = 2, padding = 'same')(x)
        x = concatenate([x, skips[i]])
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        if batch_norm:
            x = x = BatchNormalization()(x)
        x = Dropout(rate=dropout)(x)
        
    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss = [jacard_coef_loss], metrics=[jacard_coef])
    model.summary()
    return model
           
def ASPP(input_tensor, n_filters, kernel_size):
    aspp=[]
    for i in range(3):
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, dilation_rate=6*(i+1), activation='relu', padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        aspp.append(x)
        
    x = Concatenate()(aspp)
    x = Conv2D(filters=n_filters, kernel_size=1, activation='relu', padding='same')(x)
    return x
    
    
def unet_aspp(input_shape, n_filters=32, kernel_size=3, dropout=0.1, batch_norm=True, depth=4):
    skips = []
    input_layer = Input(shape=input_shape)
    x = input_layer
    # Encoder
    for i in range(depth):
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        skips.append(x)
        if batch_norm:
            x = x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(rate=dropout)(x)
        n_filters = n_filters*2
    
    # Bottleneck
    x = ASPP(x, n_filters, kernel_size=kernel_size)
    
    # Decoder
    for i in reversed(range(depth)):
        n_filters = n_filters//2
        x = Conv2DTranspose(filters=n_filters, kernel_size=3, strides = 2, padding = 'same')(x)
        x = concatenate([x, skips[i]])
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        if batch_norm:
            x = x = BatchNormalization()(x)
        x = Dropout(rate=dropout)(x)
        
    output_layer = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss = [jacard_coef_loss], metrics=[jacard_coef])
    model.summary()
    return model
#%%


# DG= ImageDataGenerator()
# train_image_generator = DG.flow_from_directory(dir_struct_dict['train_images'],batch_size=user_batch_size,seed=user_seed,target_size=user_target_size)
# train_mask_generator = DG.flow_from_directory(dir_struct_dict['train_masks'],batch_size=user_batch_size,seed=user_seed,target_size=user_target_size)
# val_image_generator = DG.flow_from_directory(dir_struct_dict['val_images'],batch_size=user_batch_size,seed=user_seed,target_size=user_target_size)
# val_mask_generator = DG.flow_from_directory(dir_struct_dict['val_masks'],batch_size=user_batch_size,seed=user_seed,target_size=user_target_size)
# test_image_generator = DG.flow_from_directory(dir_struct_dict['test_images'],batch_size=user_batch_size,seed=user_seed,target_size=user_target_size)
# test_mask_generator = DG.flow_from_directory(dir_struct_dict['test_masks'],batch_size=user_batch_size,seed=user_seed,target_size=user_target_size)

# def trainImageMaskGenerator(imageDataGenerator, maskDataGenerator):
#     combined_generator = zip(imageDataGenerator, maskDataGenerator)
#     for (img, mask) in combined_generator:
#         yield (img, mask)

# def valImageMaskGenerator(imageDataGenerator, maskDataGenerator):
#     combined_generator = zip(imageDataGenerator, maskDataGenerator)
#     for (img, mask) in combined_generator:
#         yield (img, mask)


# input_img = Input((patch_size, patch_size, nBands), name='img')
# model = get_unet(input_img, n_filters=64, dropout=0.1, batchnorm=True)
# model.compile(optimizer='adam', loss = [jacard_coef_loss], metrics=[jacard_coef])
# model.summary()

    
def train_model(STPLOT,STMSG, input_shape, depth, n_filters, kernel_size, batch_norm, dropout, user_batch_size, numEpochs, dir_struct_dict, proj_absPath, archtype):
    
    if archtype=='UNET':
        model = unet(input_shape, n_filters=n_filters, kernel_size=kernel_size, dropout=dropout, batch_norm=batch_norm, depth=depth)
    elif archtype=='UNET-Attention':
        model = unetAttn(input_shape, n_filters=n_filters, kernel_size=kernel_size, dropout=dropout, batch_norm=batch_norm, depth=depth)
    elif archtype=='UNET-AtrousSpatialPyramidPooling':
        model = unet_aspp(input_shape, n_filters=n_filters, kernel_size=kernel_size, dropout=dropout, batch_norm=batch_norm, depth=depth)
    
    # print('*****', os.path.join(dir_struct_dict['train_images']+'/train'))
    
    img_files = glob(os.path.join(dir_struct_dict['train_images']+'/train', '*.tif'))
    img_files.sort(key=lambda f: int(os.path.basename(f)[:-4].split('_')[-1]))
    
    mask_files = glob(os.path.join(dir_struct_dict['train_masks']+'/train', '*.tif'))
    mask_files.sort(key=lambda f: int(os.path.basename(f)[:-4].split('_')[-1]))
    
    # print(img_files[:5], mask_files[:5])
          
    trainGenerator = DataGenerator.Gen(img_files, mask_files, batch_size=user_batch_size)
    
    #print('***', trainGenerator.x)
    
    del img_files, mask_files
    
    img_files = glob(os.path.join(dir_struct_dict['val_images']+'/val', '*.tif'))
    img_files.sort(key=lambda f: int(os.path.basename(f)[:-4].split('_')[-1]))
    
    mask_files = glob(os.path.join(dir_struct_dict['val_masks']+'/val', '*.tif'))
    mask_files.sort(key=lambda f: int(os.path.basename(f)[:-4].split('_')[-1]))
    
    valGenerator = DataGenerator.Gen(img_files, mask_files, batch_size=user_batch_size)
    
    # print('*****', valGenerator.x)
    history = model.fit(
        trainGenerator,
        # steps_per_epoch=np.ceil(len(os.listdir(dir_struct_dict['train_images']))/user_batch_size),
        validation_data = valGenerator,
        batch_size=user_batch_size,
        # validation_steps = np.ceil(len(os.listdir(dir_struct_dict['val_images']))/user_batch_size),
        epochs = numEpochs,
        callbacks=getCallbacks(numEpochs, proj_absPath, archtype, STPLOT, STMSG),
        use_multiprocessing=False,
        verbose=1
    )
    
    with open(os.path.join(proj_absPath, 'training_history.pkl'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
