
import glob
import cv2
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from keras.utils import normalize
import tifffile as tifffile
from model import Unet
from PIL import Image


train_img_path = glob.glob('./Train/Images/*')
train_mask_path = glob.glob('./Train/Masks/*')

val_img_path = glob.glob('./Test/Images/*')
val_mask_path = glob.glob('./Test/Masks/*')

seed=24
batch_size=8

def load_img(img_list):
    images=[]
    for i in img_list:    
        img = tifffile.imread(i)
        norm_img=normalize(img, axis=1)
        images.append(norm_img)
                      
    images = np.expand_dims(np.array(images), axis=3)
    return(images)

def load_mask(mask_list):
    masks=[]
    for i in mask_list:    
        mask = tifffile.imread(i)/255.
        masks.append(mask)
                      
    masks = np.expand_dims(np.array(masks), axis=3)
    return(masks)


def ImageGenerator(img_list, mask_list, batch_size):

    L = len(img_list)

    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_list[batch_start:limit])
            Y = load_mask(mask_list[batch_start:limit])

            yield (X,Y)     

            batch_start += batch_size   
            batch_end += batch_size


train_img_gen = ImageGenerator(train_img_path,train_mask_path,batch_size)
val_img_gen = ImageGenerator(val_img_path,val_mask_path,batch_size)

steps_per_epoch = len(os.listdir('./Train/Images/'))//batch_size
val_steps_per_epoch = len(os.listdir('./Test/Images/'))//batch_size

img, mask = train_img_gen.__next__()

input_shape=(256,384,1)
model=Unet(input_shape)
model.summary()

import segmentation_models as sm
metrics = [sm.metrics.IOUScore(threshold=0.5)]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

history=model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=50,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch)

model.save('mitochondria_unet.hdf5')
