#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:01:33 2017

@author: hc
"""

import numpy as np
import keras
import pandas as pd
from scipy import misc
import CarConfig
import matplotlib.pyplot as plt
#import pydot_ng as pydot
import os

#########################
#    Global Params      #
#########################
ModelsPath = CarConfig.ModelsPath
DataPath = CarConfig.DataPath
RBG_Mode = CarConfig.RBGMode
ConvFolder2Gray = CarConfig.ConvFolder2Gray
Temporal_Buffer = CarConfig.Temporal_Buffer



#########################
#       Params          #
#########################
NumberOfFolders = 8             #Number of data folders to laod

#HYPER PARAMETERS
StopLoss = 1.4
BatchSize = 100
Epochs = 10
dropout = 0.2
dropout_thr = 0.6
optim = 'rmsprop'
LossWeights = [0.9, 0.004]


#########################
#    Local functions    #
#########################  
class LossHistory(keras.callbacks.Callback):
    ''' Keras callbacks for training interrupt on loss value '''
    global StopLoss
    
    def __init__(self):
        self.losse_old = 0
        self.endthd = 5
        self.ctr = 0
        
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.angleloss = logs.get('angle_out_loss')
        self.throttleloss = logs.get('throttle_out_loss')
        
        if (logs.get('loss') < StopLoss) and (logs.get('loss')<self.losse_old):
            self.ctr += 1
        else:
            self.ctr = max(0, self.ctr-1)
        if self.ctr > self.endthd:
            print('\n Training stopped at batch end: loss < ', StopLoss, '\n')
            self.model.stop_training = True
        self.losse_old = logs.get('loss')



def PlotLoss(loss):   
    ''' Plot training losses and save '''    
    plt.plot(loss)
    axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    axes.set_ylim([-1,100])
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title('Keras loss function')
    plt.grid(True)
    plt.savefig("Loss.png")
    plt.show()
    
    
    
def build_model(input_dim=(96, 96, 1)):
    ''' One input convolutional network '''
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers import Dropout, Flatten
    
    img_in = Input(shape=(input_dim), name='img_in')
    x = img_in
    x = Convolution2D(64, (8,8), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (4,4), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x)
    #x = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x)
    #x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    
    x = Flatten(name='flattened')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout, seed=2)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(dropout, seed=2)(x)
    x = Dense(50, activation='relu')(x)

    angle_out = Dense(1, activation='linear', name='angle_out')(x)
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
    #brake_out = Dense(1, activation='relu', name='brake_out')(x)
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    
    model.compile(optimizer=optim,
                  loss={'angle_out': 'mean_squared_error', 
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': LossWeights[0], 'throttle_out': LossWeights[1]})

    return model




def build_model_Parallel(input_dim=(96, 96, 1)):
    ''' Temporal_Buffer number of inputs convolutional network 
    - currently four'''
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.layers import Dropout, Flatten
    from keras.utils import plot_model
    
    img_in1 = Input(shape=(input_dim), name='img_in1')
    x1 = img_in1
    x1 = Convolution2D(64, (8,8), strides=(2,2), activation='relu')(x1)
    x1 = Convolution2D(64, (4,4), strides=(2,2), activation='relu')(x1)
    x1 = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x1)
    #x1 = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x1)
    #x1 = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x1)
    x1 = Flatten(name='flattened1')(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = Dropout(dropout, seed=2)(x1)
    x1 = Dense(50, activation='relu')(x1)
#    x1 = Dropout(dropout, seed=2)(x1)
#    x1 = Dense(50, activation='relu')(x1)

    img_in2 = Input(shape=(input_dim), name='img_in2')
    x2 = img_in2
    x2 = Convolution2D(64, (8,8), strides=(2,2), activation='relu')(x2)
    x2 = Convolution2D(64, (4,4), strides=(2,2), activation='relu')(x2)
    x2 = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x2)
    #x2 = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x2)
    #x2 = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x2)
    x2 = Flatten(name='flattened2')(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(dropout, seed=2)(x2)
    x2 = Dense(50, activation='relu')(x2)
#    x2 = Dropout(dropout, seed=2)(x2)
#    x2 = Dense(50, activation='relu')(x2)

    img_in3 = Input(shape=(input_dim), name='img_in3')
    x3 = img_in3
    x3 = Convolution2D(64, (8,8), strides=(2,2), activation='relu')(x3)
    x3 = Convolution2D(64, (4,4), strides=(2,2), activation='relu')(x3)
    x3 = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x3)
    #x3 = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x3)
    #x3 = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x3)
    x3 = Flatten(name='flattened3')(x3)
    x3 = Dense(256, activation='relu')(x3)
    x3 = Dropout(dropout, seed=2)(x3)
    x3 = Dense(50, activation='relu')(x3)
#    x3 = Dropout(dropout, seed=2)(x3)
#    x3 = Dense(50, activation='relu')(x3)

    img_in4 = Input(shape=(input_dim), name='img_in4')
    x4 = img_in4
    x4 = Convolution2D(64, (8,8), strides=(2,2), activation='relu')(x4)
    x4 = Convolution2D(64, (4,4), strides=(2,2), activation='relu')(x4)
    x4 = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x4)
    #x4 = Convolution2D(32, (3,3), strides=(2,2), activation='relu')(x4)
    #x4 = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x4)
    x4 = Flatten(name='flattened4')(x4)
    x4 = Dense(256, activation='relu')(x4)
    x4 = Dropout(dropout, seed=2)(x4)
    x4 = Dense(50, activation='relu')(x4)
    
#    x4 = Dropout(dropout, seed=2)(x4)
#    x4 = Dense(50, activation='relu')(x4)

    merged = keras.layers.concatenate([x1, x2, x3, x4], axis = 1)
    angle_out = Dense(1, activation='linear', name='angle_out')(merged)
    merged = Dropout(dropout_thr, seed=2)(merged)
    throttle_out = Dense(1, activation='relu', name='throttle_out')(merged)
    #brake_out = Dense(1, activation='relu', name='brake_out')(x)
    
    model = Model(inputs=[img_in1, img_in2, img_in3, img_in4], outputs=[angle_out, throttle_out])
    
    model.compile(optimizer=optim,
                  loss={'angle_out': 'mean_squared_error', 
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': LossWeights[0], 'throttle_out': LossWeights[1]})
    
    plot_model(model, to_file='model.png', show_shapes = True)
    return model




def Load_and_Wrangle():
    ''' Load and wrangle data form NumberOfFolders folders recored during 
    CarRacing_Play.py'''
    
    if not('Y1' in locals()):       #only load data if necessary
    #Y2 = np.delete(Y2, -1, 0)
        Targets = pd.read_csv(DataPath+'/CarRacing_ActionsRewards.csv')
        T_len = [len(Targets['Steering'])]
        for i in range(1, NumberOfFolders):
            Tar = pd.read_csv(DataPath+str(i)+'/CarRacing_ActionsRewards.csv') 
            T_len.append(len(Tar['Steering']))
            Targets = Targets.append(Tar)     
            
        if Temporal_Buffer > 1:
            size = len(Targets['Steering'])//Temporal_Buffer
            Y1 = np.empty([size, Temporal_Buffer])
            Y2 = np.empty([size, Temporal_Buffer])
            ctr = -1
            for i in range(len(Targets['Steering'])):
                Buf = i % Temporal_Buffer
                if Buf != 0:
                    Y1[ctr,Buf] = Targets['Steering'].iloc[i]
                    Y2[ctr,Buf] = Targets['Throttle'].iloc[i]
                else:
                    ctr = min(ctr+1, size-1)
                    Y1[ctr,0] = Targets['Steering'].iloc[i]
                    Y2[ctr,0] = Targets['Throttle'].iloc[i]
        else:
            Y1 = Targets['Steering']
            Y2 = Targets['Throttle']
                      
        
        if (RBG_Mode == False or ConvFolder2Gray == True):
            if Temporal_Buffer > 1:
                size = T_len[0]//Temporal_Buffer
                X = np.empty([size, Temporal_Buffer, 96, 96, 1])
                ctr = -1
                for i in range(T_len[0]):
                    Buf = i % Temporal_Buffer
                    if ConvFolder2Gray == True:
                        Img_tmp = CarConfig.rgb2gray(misc.imread(DataPath+"/Img"+str(i)+".png"))
                    else:
                        Img_tmp = misc.imread(DataPath+"/Img"+str(i)+".png")
                    if Buf != 0:
                        X[ctr,Buf,:,:,0] = Img_tmp
                    else:
                        ctr = min(ctr+1, size-1)
                        X[ctr,0,:,:,0] = Img_tmp        
            else:
                X = np.empty([T_len[0], 96, 96, 1])
                for i in range(T_len[0]):
                    if ConvFolder2Gray == False:
                        X[i,:,:,0] = misc.imread(DataPath+"/Img"+str(i)+".png")
                    else:
                        Xrgb = misc.imread(DataPath+"/Img"+str(i)+".png")
                        X[i,:,:,0] = CarConfig.rgb2gray(Xrgb)
        else:
            if Temporal_Buffer > 1:
                size = T_len[0]//Temporal_Buffer
                X = np.empty([size, Temporal_Buffer, 96, 96, 3])
                ctr = -1
                for i in range(T_len[0]):
                    Buf = i % Temporal_Buffer
                    Img_tmp = misc.imread(DataPath+"/Img"+str(i)+".png")
                    if Buf != 0:
                        X[ctr,Buf,:,:,:] = Img_tmp
                    else:
                        ctr = min(ctr+1, size-1)
                        X[ctr,0,:,:,:] = Img_tmp        
            else:
                X = np.empty([T_len[0], 96, 96, 3])
                for i in range(T_len[0]):
                    X[i,:,:,:] = misc.imread(DataPath+"/Img"+str(i)+".png")
            
        if NumberOfFolders > 1:
            for j in range(1, NumberOfFolders):
                if (RBG_Mode == False or ConvFolder2Gray == True):
                    if Temporal_Buffer > 1:
                        size = T_len[j]//Temporal_Buffer
                        X_t = np.empty([size, Temporal_Buffer, 96, 96, 1])
                        ctr = -1
                        for i in range(T_len[j]):
                            Buf = i % Temporal_Buffer
                            if ConvFolder2Gray == True:
                                Img_tmp = CarConfig.rgb2gray(misc.imread(DataPath+str(j)+"/Img"+str(i)+".png"))
                            else:
                                Img_tmp = misc.imread(DataPath+str(j)+"/Img"+str(i)+".png")
                            if Buf != 0:
                                X_t[ctr,Buf,:,:,0] = Img_tmp
                            else:
                                ctr = min(ctr+1, size-1)
                                X_t[ctr,0,:,:,0] = Img_tmp  
                    else:
                        X_t = np.empty([T_len[j], 96, 96, 1])
                        for i in range(T_len[j]):
                            if ConvFolder2Gray == False:
                                X_t[i,:,:,0] = misc.imread(DataPath+str(j)+"/Img"+str(i)+".png")
                            else:
                                Xrgb = misc.imread(DataPath+str(j)+"/Img"+str(i)+".png")
                                X_t[i,:,:,0] = CarConfig.rgb2gray(Xrgb)
                else:
                    if Temporal_Buffer > 1:
                        size = T_len[j]//Temporal_Buffer
                        X_t = np.empty([size, Temporal_Buffer, 96, 96, 3])
                        ctr = -1
                        for i in range(T_len[j]):
                            Buf = i % Temporal_Buffer
                            Img_tmp = misc.imread(DataPath+str(j)+"/Img"+str(i)+".png")
                            if Buf != 0:
                                X_t[ctr,Buf,:,:,:] = Img_tmp
                            else:
                                ctr = min(ctr+1, size-1)
                                X_t[ctr,0,:,:,:] = Img_tmp  
                    else:
                        X_t = np.empty([T_len[j], 96, 96, 3])
                        for i in range(T_len[j]):
                            X_t[i,:,:,:] = misc.imread(DataPath+str(j)+"/Img"+str(i)+".png")
                X = np.concatenate((X,X_t),axis=0)
            
    if ConvFolder2Gray == True:
        print('Model train: ', NumberOfFolders, ' data folders converted to gray')
    else:
        print('Model train: Color mode is RGB:', RBG_Mode)
    
    return(X, Y1, Y2)
  
 
    
def Build_Fit_Model(X, Y1, Y2):
    ''' Build CNN keras model, callbaks and train '''
    loss_ll = []  
        
    if Temporal_Buffer >1:
        if (RBG_Mode == False or ConvFolder2Gray == True):
            model = build_model_Parallel()
        else:
            model = build_model_Parallel(input_dim=(96, 96, 3))
    else:           
        if (RBG_Mode == False or ConvFolder2Gray == True):
            model = build_model()
        else:
            model = build_model(input_dim=(96, 96, 3))
#model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])

    ModelsPath_cp = ModelsPath+"Model_weights_cp.h5"
    save_best = keras.callbacks.ModelCheckpoint(ModelsPath_cp,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='min',
                                                period=Epochs)
    
    early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                           min_delta=0.001,   
                                           patience=0,
                                           verbose=1,
                                           mode='auto')
    
    history_l = LossHistory()
    
    callbacks_list = [save_best, early_stop, history_l]
    
    if Temporal_Buffer > 1:
        X1 = X[:,0,:,:,:]
        X2 = X[:,1,:,:,:]
        X3 = X[:,2,:,:,:]
        X4 = X[:,3,:,:,:] 
        y1 = np.average(Y1, axis=1)
        y2 = np.clip(np.max(Y2, axis=1), 0.25, 1)
        
        model_hist = model.fit([X1, X2, X3, X4], [y1, y2], verbose=1, 
                               batch_size=BatchSize, epochs = Epochs, 
                               callbacks=callbacks_list) 
    else:
        model_hist = model.fit(X, [Y1, Y2], verbose=1, 
                               batch_size=BatchSize, epochs = Epochs, 
                               callbacks=callbacks_list) 
            
        loss_ll.append(model_hist.history['loss'])
        
    return(model, model_hist, history_l)



#########################
#         main          #
######################### 
if __name__ == '__main__':
 
    X, Y1,Y2 = Load_and_Wrangle()    
    
    try:
        model, model_hist, history_l = Build_Fit_Model(X, Y1, Y2)
        
        if not os.path.exists(ModelsPath):
            os.makedirs(ModelsPath)  
        
        model.save(ModelsPath+"Model_weights_.h5", overwrite=True)
        print('Default model Model_weights_.h5 saved (check also callback one)')  
        
        PlotLoss(history_l.losses)
        
    except KeyboardInterrupt:
        print('User interrupt. Save model: Y or N?')
        save = input()
        if save == 'Y' or save == 'y':
            if not os.path.exists(ModelsPath):
                os.makedirs(ModelsPath)  
            
            model.save(ModelsPath+"Model_weights.h5", overwrite=True)
            print('Model Model_weights.h5 saved')
            
            PlotLoss(history_l.losses)
            
        else:
            print('Model discarded')

    