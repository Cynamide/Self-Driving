import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import backend as K
K.common.set_image_dim_ordering('th')
def alexnet(input_shape, num_classes,lr):
        
        network=tf.keras.Sequential()
        network.add(Conv2D(96, kernel_size=(3,3), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'))
        network.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None))

        network.add(Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))
        network.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)) 

        network.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        network.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
               
               
                        kernel_initializer= 'he_normal'))

        network.add(Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'))

        network.add(MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None))

        network.add(Flatten())
        network.add(Dense(4096, activation= 'relu'))
        network.add(Dropout(0.5))
        network.add(Dense(4096, activation= 'relu'))
        network.add(Dropout(0.5))
        network.add(Dense(1000, activation= 'relu'))
        network.add(Dense(num_classes, activation= 'softmax'))

        network.compile(optimizer= tf.keras.optimizers.Adam(lr),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return network
        