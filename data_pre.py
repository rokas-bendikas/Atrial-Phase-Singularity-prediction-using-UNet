#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:28:49 2020

@author: rokas
"""

from __future__ import print_function
import os
import random
import pandas as pd
import numpy as np
import pdb
from keras.preprocessing.image import ImageDataGenerator
import glob
import skimage.io as io
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re


def adjustData(img,mask):

    mask[mask > 0.074] = 1.0
    mask[mask <= 0.074] = 0.0
    
    img = img

    return (img,mask)

       
        
def dataGenerator(train_path,csv_dir,target_size,threshold=False):
    
    
    
    csv_f = os.path.join(train_path,csv_dir)
    #csv_file_list = os.listdir(csv_f)
    csv_file_list = sorted(glob.glob(''.join([csv_f,'/*.csv'])),key=lambda x:float(re.findall("(\d+)",x)[0]))
    
    batch_y = np.zeros((len(csv_file_list),target_size[0],target_size[1],1))
    
    
    i = 0
    
    for file in csv_file_list:
       
                
        csv_file_path = os.path.join(csv_f,file)
        
                
           
        csv_file = np.genfromtxt(csv_file_path, delimiter=',')
        
        if (threshold):
        
            img,csv_file = adjustData(csv_file,csv_file)
            
        
        batch_y[i,:,:,0] = csv_file
        
        i+= 1
    
        
    return batch_y
        
        
        
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_save_prefix  = "image",save_to_dir = None,target_size = (512,512),seed = 1,show_data = False,threshold=False):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    
    
    
    X=dataGenerator(train_path=train_path,csv_dir=image_folder,target_size=target_size)
    Y=dataGenerator(train_path=train_path,csv_dir=mask_folder,target_size=target_size)
   
    image_generator = image_datagen.flow(x=X, y=Y, batch_size=batch_size, shuffle=True, seed=seed, save_to_dir=save_to_dir, save_prefix=image_save_prefix)
    
    
    for (img,mask) in image_generator:
        if (threshold):
            
            img,mask = adjustData(img,mask)
                
        if (show_data):
            for i in range(img.shape[0]):
            
                plt.figure()
                plt.imshow(img[i,:,:,0])
                plt.show()
                
                plt.figure()
                plt.imshow(mask[i,:,:,0])
                plt.show()
            
           
           
        yield (img,mask)

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        np.savetxt(os.path.join(save_path,"%d_predict.csv"%i),img, delimiter=',', fmt='%f')
