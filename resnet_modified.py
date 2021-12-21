from tensorflow.keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam


def nvidia_model(input,lr):
        resnet = ResNet50(weights = 'imagenet', include_top=False, input_shape=input)
        for layer in resnet.layers[:-4]:
            layer.trainable = False
        model = Sequential()
        model.add(resnet)
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4))
        optimizer = Adam(lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
