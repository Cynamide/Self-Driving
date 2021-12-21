import numpy as np
from mss.windows import MSS as mss
import mss.tools
import cv2
import time
from KeyPress import PressKey, ReleaseKey, W, A, S, D
import os
import keyboard


directory = "C:/Users/VA/Desktop/AI_stuff/Self Driving AI(for games)/Training data/"
os.chdir(directory) 

def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

file='training_data_2.npy'
if os.path.isfile(file):
    print('file exists, loading previous data!')
    training_data=list(np.load(file))
else:
    print('File not found, starting fresh')
    training_data=[]

for i in list(range(1,4))[::-1]:
    print(i)
    time.sleep(1)

def keypress(img,counter):
    #[W,A,S,D]
    keypress=[0,0,0,0]
    if keyboard.is_pressed('s'):
        cv2.imwrite(directory+"left/train_"+str(counter)+".png",img)
    elif keyboard.is_pressed('f'):
        cv2.imwrite(directory+"right/train_"+str(counter)+".png",img)
   # elif keyboard.is_pressed('s') and keyboard.is_pressed('a'):
       # keypress[6]=1
    #elif keyboard.is_pressed('s') and keyboard.is_pressed('d'):
       # keypress[7]=1
    elif keyboard.is_pressed('e'):
       cv2.imwrite(directory+"forward/train_"+str(counter)+".png",img)
    #elif keyboard.is_pressed('a'):
        #keypress[1]=1
    elif keyboard.is_pressed('d'):
        cv2.imwrite(directory+"brake/train_"+str(counter)+".png",img)
    #elif keyboard.is_pressed('d'):
        #keypress[3]=1

def main():
  a=0
  with mss.mss() as sct:
    monitor = {"top": 280, "left": 0, "width": 640, "height": 150}
    #crop= cv2.selectROI(np.array(sct.grab(monitor)), False,False)
    counter=0
    begin_time=time.time()
    while "Screen capturing":
        counter += 1
        img = np.array(sct.grab(monitor))
        #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=gamma_trans(img,.9)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        img=cv2.resize(img,(100,100))
        keypress(img,counter)
        #training_data.append([img,key])
        if counter % 25000==0:
            print(counter)
        #if len(training_data)>=25000:
          # break
        end_time = time.time()
        #print("FPS is ",(1/(end_time-begin_time)))
        begin_time = end_time
        if keyboard.is_pressed('p'):
            print("Get_data paused!....press o to resume.")
            while True:
                if keyboard.is_pressed('o'):
                    print("Get_data resumed!")
                    break
        if keyboard.is_pressed('q'):
            break
main()