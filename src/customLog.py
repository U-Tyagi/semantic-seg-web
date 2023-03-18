#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:08:56 2023

@author: ml_ai
"""

import keras
import numpy as np
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

class TrainingPlot(keras.callbacks.Callback):
    
    def __init__(self, projAbsPath = None, stplot=None, stmsg=None):
        self.stplot = stplot
        self.stmsg = stmsg
        self.projAbsPath = projAbsPath
        
    def on_train_begin(self, logs={}):
        #Initialize the lists holding the logs
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        
    def on_epoch_end(self, epoch, logs={}):
        
        #Append to lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('jacard_coef'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_jacard_coef'))
        
        print('Epoch{}: TLoss/VLoss:{} , {} Tjcoef/Vcooef:{} , {}'.format((epoch+1),logs.get('loss'), logs.get('val_loss'), logs.get('jacard_coef'), logs.get('val_jacard_coef')))
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 0:

            N = np.arange(0, len(self.losses))
            
            # print('>>>>>>>>>>>>>>>>>>>>>>>>',self.losses)
            # You can chose the style of your preference
            # print(plt.style.available) #to see the available options
            # plt.style.use("seaborn")

            #Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch+1))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            logPath = os.path.join(self.projAbsPath,'customLog') 
            if not os.path.exists(logPath):
                os.mkdir(logPath)
            p = os.path.join(logPath, 'Epoch_{}.png'.format(epoch))
            print('Saving Loss Curve to',p)
            plt.savefig(p)
            plt.close()
            
            if(os.path.exists(p)):
                self.stplot.image(p)
            msg = 'Best Validation Accuracy: **:green[{:.2f}]** and Best Training Accuracy: **:green[{:.2f}]**'.format(max(self.val_acc),max(self.acc))
            self.stmsg.markdown(msg)
            
            accPath = os.path.join(self.projAbsPath,'acc.txt')
            acc = max(self.val_acc)*100
            print('Writing to ',accPath)
            with open(accPath,'w') as f:
                #f.write(acc)
                f.write('{:.2f}'.format(acc))
            
            
