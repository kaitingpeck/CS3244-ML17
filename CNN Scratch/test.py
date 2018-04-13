import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping

train_data_path = './dataset/training_set'
validation_data_path = './dataset/test_set'

"""
Parameters
"""
MODEL_TYPE = "CNN"
OPTIMIZER_NAME = "RMSPROP"
img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 1976
validation_steps = 551
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 6
conv2_size = 3
pool_size = 2
classes_num = 6
lr = 0.0004
epochs = 20

model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log

log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cbks = [tb_cb]
"""
"""
callbacks=cbks,
"""
"""
Set parameters for running model
"""
filepath = "checkpoint-"+ MODEL_TYPE +"-" + OPTIMIZER_NAME + "-epoch{epoch:02d}-val_acc{val_acc:.3f}.hdf5"
early_stop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, mode='max')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint, early_stop]

model.fit_generator(
    train_generator,
    steps_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')