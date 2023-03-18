# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 19:59:06 2021

@author: User
"""
import os
import streamlit as st 
import rasterio as rs
from PIL import Image
import numpy as np
from io import StringIO
import tensorflow as tf
import prediction
import utils
from glob import glob
from tempfile import NamedTemporaryFile


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
    <p>Developed by Team, SAC</p>
    </div>
    """
    return footer


def app():
    """
    Main function that contains the application for getting predictions from 
    keras based trained models.
    """
    # Get list of saved h5 models, which will be displayed in option to load.
    
    h5_abs_path = glob(os.path.join("./models/**/", '*.h5'), recursive=True)
    acc_abs_path = glob(os.path.join("./models/**/", '*.txt'), recursive=True)
    #print('Accuracies:',acc_abs_path)
    
    display_names = {}
    for i in range(len(acc_abs_path)):
        with open(acc_abs_path[i],'r') as af:
            acc = af.readline()
        display_names[os.path.basename(os.path.dirname(h5_abs_path[i]))] = acc+'%'

    
    st.title("Predict using Segmentation Models (Geospatial Hackathon 2023)")
    st.header("Trained Models")
    st.markdown(footer_markdown(),unsafe_allow_html=True)
    model_type = st.radio("Choose trained model to load...", display_names.keys())
    msg = 'Accuracy achieved during training for **:red[{}]** is **:green[{}]**'.format(model_type,display_names[model_type])
    st.markdown(msg)
    uploaded_file = st.file_uploader("Choose an image...", type="tif", accept_multiple_files=False)
    
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        image = image.resize((64,64), Image.NEAREST)
        st.image(image, caption='Uploaded Image.', use_column_width=False)
        st.write("")
        st.write("Identifying...")
        
        inImgFile = r'./in/uploaded.tif'
        with open(inImgFile, 'wb') as f:
            f.write(uploaded_file.getvalue())
            
        print(inImgFile)
        
        # Get prediction.
        maskFile = r'./out/out.tif'
        #modelpath = r"./models/{}/{}.h5".format(model_type)
        modelpath = glob(os.path.join("./models/{}/".format(model_type), '*.h5'))[0]
        user_batch_size = 4
        user_patch_size = 512
        prediction.makeprediction(inImgFile, maskFile, modelpath, user_batch_size, user_patch_size=user_patch_size)
    
            
        outFile = r'./out/out.png'
        utils.visualize_image_mask(inImgFile, maskFile, outFile)
    
        st.image(outFile)

    

if __name__=='__main__':
    app()