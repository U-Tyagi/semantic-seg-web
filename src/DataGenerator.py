#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:34:13 2023

@author: ml_ai
"""

import numpy as np
import tifffile 
from tensorflow.keras.utils import Sequence


class Gen(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.x)/self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
        return np.array([tifffile.imread(file_name_x) for file_name_x in batch_x]), np.array([tifffile.imread(file_name_y) for file_name_y in batch_y])
        