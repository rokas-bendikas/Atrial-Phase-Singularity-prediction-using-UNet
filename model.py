import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from tensorflow.keras.metrics import KLDivergence
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def sensitivity(y_true, y_pred):
    true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.sum(keras.round(keras.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + keras.epsilon())

def specificity(y_true, y_pred):
    true_negatives = keras.sum(keras.round(keras.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = keras.sum(keras.round(keras.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + keras.epsilon())



def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = keras.sum(keras.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (keras.sum(keras.square(y_true),-1) + keras.sum(keras.square(y_pred),-1) + smooth)
    
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    
    
def tversky(y_true, y_pred,alpha = 0.7,smooth=1):
    y_true_pos = keras.flatten(y_true)
    y_pred_pos = keras.flatten(y_pred)
    true_pos = keras.sum(y_true_pos * y_pred_pos)
    false_neg = keras.sum(y_true_pos * (1-y_pred_pos))
    false_pos = keras.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred,alpha = 0.3):
    return 1 - tversky(y_true,y_pred,alpha)
    
    


def unet(pretrained_weights = None,input_size = (512,512,1),loss='binary_crossentropy'):
    inputs = Input(input_size)
    
    
    k_init = 'he_normal'
    #k_init = 'Zeros'
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = k_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = k_init)(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = SGD(momentum=0.9), loss = loss, metrics = ['accuracy',sensitivity,specificity])
    model.summary()
    

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

