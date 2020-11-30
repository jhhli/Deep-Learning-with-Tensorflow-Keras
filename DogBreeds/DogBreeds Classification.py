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

# compile model
from keras import optimizers

VGG_16 = Sequential()
VGG_16.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


VGG_16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
VGG_16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

VGG_16.add(Flatten())

VGG_16.add(Dense(1024,activation="relu"))
VGG_16.add(Dense(512,activation="relu"))
VGG_16.add(Dense(128,activation="relu"))


VGG_16.add(Dense(23, activation="softmax"))

VGG_16.summary()

# Compile model
VGG_16.compile(loss='categorical_crossentropy',
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

history = VGG_16.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs,
    callbacks = [checkpoint, early_stop, learning_rate_reduction])














