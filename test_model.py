import numpy as np
from mss.windows import MSS as mss
import mss.tools
import cv2
import time
from KeyPress import PressKey, ReleaseKey, W, A, S, D
import os
import keyboard
from alexnetTFv2 import alexnet
import tensorflow as tf

model_name='GTA-SA-0.0001-AlexNetv2-2-epochs.model'
height = 100
width = 100
def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

def forward():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def brake():
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)
    
def left():
    ReleaseKey(D)
    PressKey(A)
    #PressKey(W)
    ReleaseKey(S)
    

def right():
    ReleaseKey(A)
    PressKey(D)
    ReleaseKey(S)
    
    
    

model= tf.saved_model.load("GTA-SA-0.001-AlexNetv2-8-epochs.model")    

for i in list(range(1,4))[::-1]:
    print(i)
    time.sleep(1)

last_time = time.time()

with mss.mss() as sct:
    monitor = {"top": 280, "left": 0, "width": 640, "height": 150}
    #crop= cv2.selectROI(np.array(sct.grab(monitor)), False,False)
    pause=False
    while "Screen capturing":
      if not pause:  
        last_time = time.time()
        time.sleep(0.07)
        img = np.array(sct.grab(monitor))
        img=gamma_trans(img,.9)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img=cv2.resize(img,(100,100))
        print("fps: {}".format(1 / (time.time() - last_time)))
        last_time = time.time()
        prediction = model([img.reshape(width,height,3)])[0]
        decoded = np.argsort(prediction)
        move = decoded[2]
        #prediction = model.predict([img.reshape(1,width,height,1)])
        #print(moves,prediction)
        #print(move,prediction)
        if move == 9:
            pass
            #brake()
        elif move ==  0:
            forward()
        elif move == 1:
            left()
        elif move == 2:
            right()
        if keyboard.is_pressed('q'):
            print("Paused")
            pause=True
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            ReleaseKey(S)
            time.sleep(1)
            while pause:
             if keyboard.is_pressed('q'):
                pause=False
                time.sleep(1)
                break
        
        if keyboard.is_pressed('e'):
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            ReleaseKey(S)
            time.sleep(1)
            break
        