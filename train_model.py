import numpy as np
from alexnetTFv2 import alexnet
import cv2
from tensorflow.keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
input=(84,84,1)
width=84
height=84
lr=1e-3
epoch=8
model_name='just-cause-AI-{}-{}-{}-epochs.model'.format(lr,"AlexNetv2",epoch)
train_dir = 'C:/Users/VA/Desktop/AI_stuff/Self Driving AI(for games)/Training data balanced'

resnet = ResNet50(weights = 'imagenet', include_top=False, input_shape=(84,84,3))
for layer in resnet.layers[:-4]:
    layer.trainable = False

def nvidia_model():
  model = Sequential()
  model.add(resnet)
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  optimizer = Adam(1e-3)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

model = nvidia_model()
print(model.summary())


datagen = ImageDataGenerator(validation_split = 0.2,rescale=1./255)
train_gen = datagen.flow_from_directory(
    train_dir,
    class_mode="categorical",
    subset='training',
    target_size=(84,84,3),
    batch_size=32)
val_gen = datagen.flow_from_directory(
    train_dir,
    subset='validation',
    class_mode="categorical",
    target_size=(84,84,3),
    batch_size=32)

history = model.fit(train_gen,
          epochs = epoch,
          steps_per_epoch = 1250, 
          validation_data = val_gen,
          validation_steps = 150)
          
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
# tensorboard --logdir=foo:C:/Users/VA/Desktop/AI stuff/log
model.save(model_name)














"""train_data = np.load("final_data_basic_control(GTA).npy")
train = train_data[:-400]
test = train_data[-400:]
X = np.array([i[0] for i in train]).reshape(-1,height, width, 1)
cv2.imshow("window",X[0])
print(np.shape(X))
Y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1,width, height, 1)
test_Y = [i[1] for i in test]"""
