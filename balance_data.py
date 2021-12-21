import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import h5py
import os
import random
import shutil

train_folder = "C:/Users/VA/Desktop/AI_stuff/Self Driving AI(for games)/Training data/"
balanced_train_folder =  "C:/Users/VA/Desktop/AI_stuff/Self Driving AI(for games)/Training data balanced/"

forward = os.listdir(train_folder+"forward") 
forward = random.sample(forward, 8000)
left = os.listdir(train_folder+"left") 
left = random.sample(left, 8000)
right = os.listdir(train_folder+"right") 
right = random.sample(right, 8000)
brake = os.listdir(train_folder+"brake") 

for file in forward:
  shutil.copy(train_folder+"forward/"+str(file),balanced_train_folder+ "forward/"+str(file) )
  os.remove(train_folder+"forward/"+str(file))

for file in left:
  shutil.copy(train_folder+"left/"+str(file),balanced_train_folder+ "left/"+str(file) )
  os.remove(train_folder+"left/"+str(file))

for file in right:
  shutil.copy(train_folder+"right/"+str(file),balanced_train_folder+ "right/"+str(file) )
  os.remove(train_folder+"right/"+str(file))

for file in brake:
  shutil.copy(train_folder+"brake/"+str(file),balanced_train_folder+ "brake/"+str(file) )
  os.remove(train_folder+"brake/"+str(file))















"""
#train_data_1=np.load('training_data_1.npy')
train_data=np.load('training_data_1.npy')

delete=[]
#for i in range(len(train_data)):
 #   if train_data[i,1]==[1, 1, 0, 1]:
 #     delete.append(i)
 #   elif train_data[i,1]==[0, 1, 0, 1]:
 #     delete.append(i)

#train_data=np.delete(train_data,delete,axis=0)
#np.save('training_data.npy',train_data)
#df=pd.DataFrame(train_data_1)
#print(Counter(df[1].apply(str)))
#df=pd.DataFrame(train_data_2)
#print(Counter(df[1].apply(str)))
left=[]
right=[]
forward=[]
brake=[]
do_nothing=[]
forward_left=[]
forward_right=[]
for data in train_data:
#[W,A,S,D,WA,WD,SA,SD]
    img = data[0]
    key = data[1] 
    if key == [1, 0, 0, 0, 0, 0, 0, 0]:
      np.delete(key,  np.s_[6:8], axis=1)
      forward.append([img,key])
    elif key == [0, 1, 0, 0, 0, 0, 0, 0]:
      np.delete(key,  np.s_[6:8], axis=1)
      left.append([img,key])
    elif key == [0, 0, 1, 0, 0, 0, 0, 0]:
      np.delete(key,  np.s_[6:8], axis=1)
      brake.append([img,key])
    elif key == [0, 0, 0, 1, 0, 0, 0, 0]:
      np.delete(key,  np.s_[6:8], axis=1)
      right.append([img,key])
    elif key == [0, 0, 0, 0, 1, 0, 0, 0]:
      np.delete(key,  np.s_[6:8], axis=1)
      forward_left.append([img,key])
    elif key == [0, 0, 0, 0, 0, 1, 0, 0]:
      np.delete(key,  np.s_[6:8], axis=1)
      forward_right.append([img,key])
    elif key == [0, 0, 0, 0, 0, 0, 0, 0]:
      np.delete(key,  np.s_[6:8], axis=1)
      do_nothing.append([img,key])
    
    else:
      print("no match!!")

shuffle(forward)
forward = forward[:2500]
#do_nothing = do_nothing[-4500:]
#forward_lefts = forward_lefts[-4500:]
#forward_rights = forward_rights[-4500:]
#rights = rights[-7300:]
#lefts = lefts[-7300:]
#brakes=brakes[:3120]

basic_control_120x120_1= forward + left + right +forward_left + forward_right +do_nothing +brake 

shuffle(basic_control_120x120_1)

print(np.shape(basic_control_120x120_1))


np.save('basic_control_120x120_1.npy',basic_control_120x120_1)"""