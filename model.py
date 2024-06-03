import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau

def unet(input_size = (128,128,3)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(bn1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(bn2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(bn3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(bn4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(bn5)
    conv5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv6 = Conv2D(2048, 3, activation = 'relu', padding = 'same')(pool5)
    bn6 = BatchNormalization()(conv6)
    conv6 = Conv2D(2048, 3, activation = 'relu', padding = 'same')(bn6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = concatenate([Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(conv6), conv5], axis=3)
    conv7 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(up7)
    bn7 = BatchNormalization()(conv7)
    conv7 = Conv2D(1024, 3, activation = 'relu', padding = 'same')(bn7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv7), conv4], axis=3)
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same')(up8)
    bn8 = BatchNormalization()(conv8)
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same')(bn8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv8), conv3], axis=3)
    conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same')(up9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same')(bn9)
    conv9 = BatchNormalization()(conv9)

    up10 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv9), conv2], axis=3)
    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same')(up10)
    bn10 = BatchNormalization()(conv10)
    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same')(bn10)
    conv10 = BatchNormalization()(conv10)

    up11 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv10), conv1], axis=3)
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same')(up11)
    bn11 = BatchNormalization()(conv11)
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same')(bn11)
    conv11 = BatchNormalization()(conv11)

    conv11 = Conv2D(3, 1, activation = 'sigmoid', padding = 'same')(conv11)
 
    model = Model(inputs, conv11)

    model.compile(optimizer = 'Adam', loss = 'mse')

    return model


