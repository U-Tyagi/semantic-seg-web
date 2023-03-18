# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 21:30:54 2021

@author: User
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import streamlit as st
import constants as C
import os
import shutil
import data_preparation
import train_model

def footer_markdown():
    footer="""
    <style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }
    
    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }
    .css-91z34k {
    width: 100%;
    padding-left: 1rem;
    padding-right: 1rem;
    padding-top: 2rem;
    padding-bottom: 10rem;
    max-width: 46rem;
    }
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed by Team SAC</p>
    </div>
    """
    return footer

def app():
    print('\n---------Running App from start---------------')
    st.title("Image Segmentation (Geospatial Hackathon 2023)")
    st.header("Make/Train your Model")
    st.subheader("Create Network:")
    st.markdown(footer_markdown(),unsafe_allow_html=True)
    
    proj_dir = C.projDir
    origImageDir = C.origImageDir
    
    with st.form(key = "form"):
        proj_name = st.text_input("Enter Project Name")
        input_shape = st.text_input("Enter input shape as height,width,channels for UNET",key="999")
        
        uploaded_file = st.file_uploader("Choose an image...", type="tif", accept_multiple_files=True)
        del uploaded_file
        
        archtype = st.radio("Select Architecture", ["UNET", "UNET-Attention","UNET-AtrousSpatialPyramidPooling"], index=0, horizontal=True)
        
        # Architecture Parameters
        depth = st.number_input("Enter Depth in UNET",min_value=3, max_value=5, value=4, step=1)
        n_filters = st.number_input("Enter number of Features/Filters",min_value=32, max_value=64, value=32, step=16)
        kernel_size = st.number_input("Enter Kernel size in Integer",min_value=3, max_value=7, value=3, step=2)
        batch_norm = st.radio(" Apply Batch Normalization?", ["True", "False"], index=0, horizontal=True)
        dropout = st.number_input("Enter Dropout Rate",min_value=0.1, max_value=0.5, value=0.2, step=0.1)
        user_batch_size = st.number_input("Enter Batch Size",min_value=4, max_value=64, value=4, step=4)
        
        
        # Training Parameters
        val_ratio = st.number_input("Enter Validation Ratio", min_value=0.1, max_value=0.9, value=0.1, step=0.1)
        numEpochs = st.number_input("Enter number of Epochs",min_value=5, max_value=100, value=20, step=5)
        
        train_button = st.form_submit_button(label="Train Data")
    
    if train_button:
        if proj_name:
            print('Project Name is supplied, initiate check.',proj_name)
            proj_name = proj_name+'_D'+str(depth)+'_F'+str(n_filters)+'_'+archtype
            proj_absPath = os.path.join(proj_dir,proj_name)
            trainingDir = os.path.join(proj_absPath,'trainingdata')
            aug_dir = os.path.join(proj_absPath, 'augData')
            
            images_dir = os.path.join(aug_dir, 'images')
            masks_dir = os.path.join(aug_dir, 'masks')
            
            dir_struct_dict = {'train_images': os.path.join(proj_absPath + '/splitData', 'train_images'),
                          'test_images' : os.path.join(proj_absPath + '/splitData', 'test_images'),
                          'val_images' : os.path.join(proj_absPath + '/splitData', 'val_images'),
                          'train_masks': os.path.join(proj_absPath + '/splitData', 'train_masks'),
                          'test_masks': os.path.join(proj_absPath + '/splitData', 'test_masks'),
                          'val_masks': os.path.join(proj_absPath + '/splitData', 'val_masks')}
            
            if os.path.isdir(proj_absPath):
                st.write(":red[Project Name already exists, You need to refresh and give inputs again!]")
                print('PROJNAME Check Failed STOP!')
                st.stop()
                    
            print('Input_shape',input_shape)
            # Input2
            if input_shape:
                print('Input Shape is supplied, initiate check.',input_shape)
                input_shape=tuple([int(i) for i in input_shape.split(",")])
        
                if not isinstance(input_shape, tuple):
                    st.write("Enter valid input shape, You need to refresh and give inputs again!")
                    print('Input Shape Check Failed STOP!')
                    st.stop()

                user_patch_size = input_shape[0]
                
                # Input3 Checks
                #                
                #
                val_ratio = 1 - float(val_ratio)
                user_batch_size = int(user_batch_size)
                numEpochs = int(numEpochs)
                n_filters = int(n_filters)
                user_seed=11
                
                #TRAIN CALL / LOGIC EXECUTION
                print("BEST PLACE TO COPY TRAINING DATA & CALL")
                print('Making Dirs')
                if not os.path.isdir(proj_absPath):    
                    os.mkdir(proj_absPath)
                if not os.path.isdir(trainingDir):
                    os.mkdir(trainingDir)
                    
                # Copy files from client side to server side
                # for f in os.listdir(origImageDir):
                #    shutil.copy2(os.path.join(origImageDir, f), trainingDir)
                
                
                if os.path.exists(trainingDir):
                    ST_MSG = st.empty()
                    data_preparation.prepare_data(ST_MSG, trainingDir, images_dir, masks_dir, dir_struct_dict, user_seed, user_patch_size, ttv=None, validation_ratio=val_ratio, aug=True,split=True)
                    
                    st.markdown("**:green[Training .....]**")
                    STPLOT = st.empty()
                    STMSG = st.empty()
                    train_model.train_model(STPLOT,STMSG, input_shape, depth, n_filters, kernel_size, batch_norm, dropout, user_batch_size, numEpochs, dir_struct_dict, proj_absPath, archtype)
                    
                    st.markdown("**:green[Training Complete!]**")
                    print('[Training Complete!]')
            else:
                st.markdown(":red[Input Shape not supplied!]")
                print('Input Shape Check Failed STOP!')
                st.stop()
                    
        else:
            print('Project Name is not supplied')
            st.markdown(':red[MUST SUPPLY UNIQUE PROJ NAME]')
            st.stop()
    else:
        print('Train button not clicked!')

def app1():
    """
    Main function that contains the application to train keras based models.
    """
    print('Running App')
    st.title("Keras Training Basic UI")
    st.header("A Streamlit based Web UI To Create And Train Models")
    st.subheader("Create Network:")
    st.markdown(footer_markdown(),unsafe_allow_html=True)
    
    msg = st.empty()
    proj_dir = C.projDir
    origImageDir = C.origImageDir
    
    proj_name = st.text_input("Enter Project Name")
    if proj_name:
        proj_absPath = os.path.join(proj_dir,proj_name)
        trainingDir = os.path.join(proj_absPath,'trainingdata')
        aug_dir = os.path.join(proj_absPath, 'augData')
        
        images_dir = os.path.join(aug_dir, 'images')
        masks_dir = os.path.join(aug_dir, 'masks')
        
        dir_struct_dict = {'train_images': os.path.join(proj_absPath + '/splitData', 'train_images'),
                      'test_images' : os.path.join(proj_absPath + '/splitData', 'test_images'),
                      'val_images' : os.path.join(proj_absPath + '/splitData', 'val_images'),
                      'train_masks': os.path.join(proj_absPath + '/splitData', 'train_masks'),
                      'test_masks': os.path.join(proj_absPath + '/splitData', 'test_masks'),
                      'val_masks': os.path.join(proj_absPath + '/splitData', 'val_masks')}
        
        user_seed=11
        
        print('TD',trainingDir)
        print('PD',proj_absPath)    
        
        if os.path.isdir(proj_absPath):
            msg.write("Project Name already exists")
            st.stop()
        else:
            print('Making Dirs')
            if not os.path.isdir(proj_absPath):    
                os.mkdir(proj_absPath)
            if not os.path.isdir(trainingDir):
                os.mkdir(trainingDir)
            
            # clientFiles = st.file_uploader("Choose Training Data", type="tif", accept_multiple_files=True)
            
        for f in os.listdir(origImageDir):
            shutil.copy2(os.path.join(origImageDir, f), trainingDir)

        in_pl = st.empty()
        input_shape = in_pl.text_input("Enter input shape height,width,channels",key="999")
        if input_shape:
            input_valid = False
            input_shape=tuple([int(i) for i in input_shape.split(",")])
    
            placeholder = st.empty()
            if isinstance(input_shape, tuple):
                input_valid = True
            else:
                input_shape = in_pl.text_input("Enter input shape as tuple",key="998")
                placeholder.write("Invalid input shape.")
                
            user_patch_size = input_shape[0]
                
            arch = st.radio("Choose Architecture",("UNET","UNET-ATTN"))
            depth = st.number_input("Enter Model Depth")
            n_filter = st.number_input("Enter Number of filters")
            kernel_size = st.number_input("Enter Kernel Size")
            dropout = st.number_input("Enter DROPOUT rate")
            batchnorm = st.radio("Batch Normalisation?",("True","False"))
            
            batch_size = st.number_input("Enter Batch Size")
            val_ratio = st.number_input("Enter Validation Ratio")
            epochs = st.number_input("Enter number of Epochs")
            
            # Load data.
            # trainingDir = st.text_input("Enter training dataset directory")
            
            save_model = st.text_input("Model name, if want to save model...")
            
            val_ratio = float(val_ratio)
            batch_size = int(batch_size)
            epochs = int(epochs)
            
            
        
                            
            # # model = keras.Model(inputs=inputs, outputs=outputs)
            
            # if save_model:
            #     save_model = save_model + '_' + arch
            #     save_condition = st.radio("Choose save condition...",
            #                               ("train acc","val acc","train loss","val loss"))
                
            if st.button("Train"):
                st.write("Starting training with {} epochs...".format(epochs))
                
                if os.path.exists(trainingDir):
                    data_preparation.prepare_data(trainingDir, images_dir, masks_dir, dir_struct_dict, user_seed, user_patch_size, ttv=None, validation_ratio=val_ratio, aug=True,split=True)
                        
                        
                
                
                                            
                                                                
                                                    
if __name__=='__main__':
    app()
