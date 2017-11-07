#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:55:26 2017

@author: hc
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:55:38 2017

@author: hc

Simulation after CarRacing_Play.py and CarRacing_Learn.py
"""      

#To generate video:
#ffmpeg -f image2 -r 6 -i 'Img%01d.png' output.mp4


import gym
import numpy as np
from keras.models import load_model
import CarConfig



#########################
#    Global Params      #
#########################
ModelsPath = CarConfig.ModelsPath
RBG_Mode = CarConfig.RBGMode
ConvFolder2Gray = CarConfig.ConvFolder2Gray
Temporal_Buffer = CarConfig.Temporal_Buffer



#########################
#       Params          #
#########################
DataPath = 'data/imitate'
StoreData = False       #Store simulation outputs in data/imitate
max_episodes = 4
max_steps = 900000


#########################
#    Local functions    #
#########################  

class Imitation():
    ''' Handles buffer of environement states for simulation '''
    def __init__(self, model, state):
        if RBG_Mode == False or ConvFolder2Gray==True:
            self.x4 = self.x3 = self.x2 = self.x1 = np.empty([1, 96, 96, 1])
            self.x = self.x1
        else:
            self.x4 = self.x3 = self.x2 = self.x1 = np.empty([1, 96, 96, 3])
            self.x = self.x1
            
        self.model = model
        self.state = state
        
    def update_states(self, state):

        if Temporal_Buffer > 1:
            if RBG_Mode == False or ConvFolder2Gray==True:
                s = CarConfig.rgb2gray(state)
                self.x4 = self.x3
                self.x3 = self.x2
                self.x2 = self.x1
                self.x1[0,:,:,0] = s
            else:
                self.x4 = self.x3
                self.x3 = self.x2
                self.x2 = self.x1
                self.x1[0,:,:,:] = state
            
            return self.x1, self.x2, self.x3, self.x4
       
        else:
            if RBG_Mode == False or ConvFolder2Gray==True:
                self.x[0,:,:,0] = CarConfig.rgb2gray(state)
            else:
                self.x[0,:,:,:] = state
                
            return self.x
   
    
    def Predict_Simulate(self):
        ''' Predict actions based on trained model in CarRacing_Learn.py '''
        reward_l = []
        action_l = []
        state_l = []
        done = False
        reward_total = 0
        
        try: 
            state = self.state                     
            for step in range(max_steps):
                if(step == (max_steps-1)):
                        print ('Max number of steps reached')
                    
                env.render(mode = 'human')  
    
                if Temporal_Buffer > 1:
                    x1, x2, x3, x4 = self.update_states(state) 
                    act = self.model.predict([x1,x2,x3,x4])            
                else:
                    x = self.update_statesupdate(state) 
                    act = self.model.predict(x)
                    
                act[1][0][0] = max(act[1][0][0], 0.02) #clip speed  
                
                if (step % 10)==0:
                    print('Angle: ', act[0][0][0], 'Throttle: ', act[1][0][0])   
                
                done_old= done
                state, reward, done, _ = env.step(np.append(act, 0))
                reward_l.append(reward)
                action_l.append(np.append(act, 0))
                state_l.append(state)
        
                #Signal that episode is done but don't interrupt simulation
                if done and not(done_old):
                    reward_total = sum(reward_l)
                    print("Done!")
                    #break
            
        except KeyboardInterrupt:
            if StoreData:
                CarConfig.save_data(DataPath, action_l, reward_l, state_l)
                print("User interrupt, simulation data saved")
            else:
                print("User interrupt")
                
        return action_l, state_l, reward_l, reward_total




#########################
#         main          #
######################### 
if __name__ == '__main__':

    env = gym.make('CarRacing-v0')
    env.seed(2)
    
    from gym import envs
    envs.box2d.car_racing.WINDOW_H = 500
    envs.box2d.car_racing.WINDOW_W = 600

    model = load_model(ModelsPath+"Model_weights_.h5")
        
    state = env.reset()
    Imitate = Imitation(model, state)
    
    action_l, state_l, reward_l, reward_total = Imitate.Predict_Simulate()
     
    if StoreData:
        CarConfig.save_data(DataPath, action_l, reward_l, state_l)     
        print("Simulation data will saved")         

    env.close()        

