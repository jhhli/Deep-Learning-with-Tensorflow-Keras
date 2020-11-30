# this code use VGG16 model with its weights in convolutional layers

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import tensorflow as tf
train_data_directory = 'C:/Users/Jenny/Desktop/images'

# image data preprocessing
image_size = (224,224)

batch_size = 32

class_mode = 'categorical' 
# for multi-class classification problem, use: class_mode = 'categorical' 
# for binary classification problem, use: class_mode = 'binary' 

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2 # set validation split
    ) 

train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode, # for multi-class classification problem, use 'category'
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_directory, # same directory as training data
    target_size=image_size,
    batch_size=batch_size,
    class_mode=class_mode, # for multi-class classification problem, use 'category'
    subset='validation') # set as validation data

# build a model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization, Dropout
from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
from keras import Model

model = ResNet50(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)

for layer in model.layers:
    layer.trainable = False

last = model.output

x = Flatten()(last)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(23, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=model.input, outputs=predictions)

model.summary()

# compile model
from keras import optimizers
model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.Adam(lr=0.001),
          metrics=['acc'])

# Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
checkpoint = ModelCheckpoint("dogbreeds_model.h5", 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             save_weights_only=False, 
                             mode='auto')

early_stop = EarlyStopping(monitor='val_acc', 
                      min_delta=0, 
                      patience=10, 
                      verbose=1, 
                      mode='auto')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.0000001)

# fit/train model

nb_epochs = 20

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs,
    callbacks = [checkpoint, early_stop, learning_rate_reduction])

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

# load an image from file
image = load_img('C:/Users/Jiahu/Desktop/images/IMG-7629.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)

import numpy as np
yhat = model.predict(image)
print(list(train_generator.class_indices.keys())[list(train_generator.class_indices.values()).index(np.argmax(yhat[0]))]) 
print(np.max(yhat[0]*100), '%')