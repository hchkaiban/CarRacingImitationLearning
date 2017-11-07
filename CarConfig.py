#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:07:28 2017

@author: hc

Global config 
"""
import shutil, os
import numpy as np
import pandas as pd
from scipy import misc

ModelsPath = "KerasModels/"
DataPath = "data/play"
RBGMode = True          #If false, recorded data shall be Gray
ConvFolder2Gray = False #If True, recorded RGB data is converted to Gray

#Number of images to stack and train (memory)
#Shall be Either 4 or 0 (1 same as 0) else keras model in CarRacing_Learn shall be updated either
Temporal_Buffer = 4 



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def save_data(Path, action_l, reward_l, state_l):
    ''' Saves actions, rewards and states (images) in DataPath'''
    if not os.path.exists(Path):
        os.makedirs(Path)
    else:
        shutil.rmtree(Path)
        os.makedirs(Path)
    
    df = pd.DataFrame(action_l, columns=["Steering", "Throttle", "Brake"])
    df["Reward"] = reward_l
    df.to_csv(Path+'/CarRacing_ActionsRewards.csv', index=False)
    
    #img = np.empty(len(state_l))
    for i in range(len(state_l)):
        if RBGMode == False:
            image = rgb2gray(state_l[i])
        else:
            image = state_l[i]
        ##misc.imsave("d7ata/Img"+str(i)+".png", state_l[i])
        misc.imsave(Path+"/Img"+str(i)+".png", image)