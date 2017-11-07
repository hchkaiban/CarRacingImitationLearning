#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:55:38 2017

@author: hc

Play and record data for imitation learning: see CarRacing_Learn.py 
for training and CarRacing_Imitate.py for simulating.

CarRacing-v0:
Actions
    Steering: Real valued in [-1, 1]
    Gas: Real valued in [0, 1]
    Break: Real valued in [0, 1]
Observations:
   STATE_W = 96  * STATE_H = 96 * RGB   
   For RLL coudl be simplified in velocity, angle, and distance from center
   see https://github.com/openai/gym/issues/524   
Reward: 
    -0.1 every frame and +1000/N for every track tile visited, where N is the 
    total number of tiles in track. For example, if you have finished in 732 frames, your
    reward is 1000 - 0.1*732 = 926.8 points. Episode finishes when all tiles are visited. 
    Some indicators shown at the bottom of the window and the state RGB buffer. From left 
    to right: true speed, four ABS sensors, steering wheel position, gyroscope. 
"""  
       
import gym
import numpy as np
import pandas as pd
#import threading
#import signal
#from PIL import Image
from scipy import misc
import shutil, os
import CarConfig

#########################
#       Params          #
#########################
RBG_Mode = CarConfig.RBGMode    #Save images in RBG else gray scale
DataPath = CarConfig.DataPath   
ClearRecordings = True          #Clear existing recordings before starting new ones  

#max_episodes = 4
max_steps = 10000               #Number of game steps before saving data (on keyboard interrupt data are also saved)

#Discretized actions, enter raw index od selected action in console
action_buf_str =    ['None', 'Brake', 'SharpLeft', 'SlightLeft', 'Staight','SlightRight', 'SharpRight' ]   
action_buffer = np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.6],
                          [-0.3, 0.05, 0.05],
                          [-0.1, 0.1, 0.0],
                          [0.0, 0.2, 0.0],
                          [0.1, 0.1, 0.0],   
                          [0.4, 0.05, 0.0]   ])
NumberOfDiscActions = len(action_buffer)
  


#########################
#    Local functions    #
#########################  
def get_action():
    ''' gets action act from console '''
    
    print('Enter action')
    act = input()
    
    try:
        #check if input is an integer
        act = int(act) 
    except:
        print(act, "not an integer or >=", NumberOfDiscActions)
        act = 0
        get_action()
    
    act = min(act,NumberOfDiscActions-1)
    disc_act = action_buffer[act]
    disc_str = action_buf_str[act]
    return (disc_act, disc_str)
    
    print('except next')    



def slew_rate(rate, a, rst=False):
    ''' Continuously change action a with slope rate'''
    signs = [0,0,0]
    
    if not hasattr(slew_rate, "end"):
        slew_rate.end = [0,0,0]  # it doesn't exist yet, so initialize it
    if not hasattr(slew_rate, "a_t"):
        slew_rate.a_t = [0,0,0]  # it doesn't exist yet, so initialize it
        
    if rst==True:
        #slew_rate.a_t = a_old  # it doesn't exist yet, so initialize it
        slew_rate.end = [0,0,0]
        
    for i in range(len(a)):
        if a[i] >= slew_rate.a_t[i]:
            signs[i] = 1
        else:
            signs[i] = -1
    if sum(slew_rate.end)<3:
        for i in range(len(a)):
            slew_rate.a_t[i] += signs[i]*rate
            if (signs[i] * (slew_rate.a_t[i] - a[i]) >= 0):
                slew_rate.end[i] = 1
     
    return slew_rate.a_t 



def run(env):
    ''' Play the game!'''
    rew_t = 0
    rew_l = []
    act_l = []
    sta_l = []
    
#    for episode in range(max_episodes):       
    env.reset()
    try:
        for step in range(max_steps):
            if(step == (max_steps-1)):
                print ('Max number of steps reached')
                
            env.render(mode = 'human')  
            
            reset = False
            if (step % 10)==0:
                action, action_str = get_action()
                reset = True
            
            action_slew = slew_rate(0.02, action, reset) 
            print(action_str, ':', action_slew)                           
            
            state, reward, done, _ = env.step(np.array(action_slew))
            
            rew_t += reward
            print("step:", step, 'R: %.6f' % rew_t, done)
            
            rew_l.append(reward)
            act_l.append(np.asarray(action_slew))
            sta_l.append(state)
        
    except KeyboardInterrupt:
        CarConfig.save_data(DataPath, act_l, rew_l, sta_l)
        print("User interrupt, data saved")
        
    return  rew_t, rew_l, act_l, sta_l 
      
#            if done:
#                print('Episode : ', episode, 'done.')
#                break


   
    
#########################
#         main          #
#########################     
if __name__ == '__main__':

    env = gym.make('CarRacing-v0')
    env.seed(2)
    from gym import envs
    envs.box2d.car_racing.WINDOW_H = 500
    envs.box2d.car_racing.WINDOW_W = 600
    
    reward_t, reward_l, action_l, state_l = run(env)
    reward_total = sum(reward_l)
    
    CarConfig.save_data(DataPath, action_l, reward_l, state_l)
    print("Data saved")
    
    env.close()
