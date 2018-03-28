import os, os.path
import glob
import shutil
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
 
run_idx = 1
model_type = "VGG16"
optimizer_name = "RMSProp"
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_DEPTH = 3 #RGB

# Set parameters
root_dir = 'C:/Users/Kai/Desktop/CS3244/Project/data/test-runs/' + str(run_idx)
src_dir_all_data = 'C:/Users/Kai/Desktop/CS3244/Project/data/dataset-resized/'
item_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
num_classes = len(item_types)
batch_size = 20
num_epochs = 20

# Data pre-processing - split into training and validation set (using k as the ratio)
def split_data(root_dir, k):

    nTrain, nVal = 0, 0
    
    # Create train and validation folders
    train_dir = root_dir + '/train' 
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    val_dir = root_dir + '/val' 
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # For each item, copy the training and test data into a new class file
    for item in item_types:
        src_dir = src_dir_all_data + item
        files = next(os.walk(src_dir))[2]
        num_files = len(files)

        num_train = round(k*num_files)
        num_val = num_files - num_train
        nTrain += num_train
        nVal += num_val

        # Create folder in train for this class
        train_dir_class = train_dir +  '/' + item 
        if not os.path.exists(train_dir_class):
            os.makedirs(train_dir_class)

        # Create folder in val for this class
        val_dir_class = val_dir + '/' + item 
        if not os.path.exists(val_dir_class):
            os.makedirs(val_dir_class)

        # Copy files to new folder
        idx = 0
        for file in glob.iglob(os.path.join(src_dir, "*.jpg")):
            if idx < num_train:
                shutil.copy2(file, train_dir_class)
            else:
                shutil.copy2(file, val_dir_class)
            idx += 1

    return train_dir, val_dir, nTrain, nVal

# Load pre-trained model
def load_model():
     conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH))
     return conv_base

def extract_features_train(conv_base, train_dir, nTrain, batch_size, num_classes):
    datagen = ImageDataGenerator(rescale=1./255)
     
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(INPUT_WIDTH, INPUT_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    return train_generator

def extract_features_val(conv_base, val_dir, nVal, batch_size, num_classes):
    datagen = ImageDataGenerator(rescale=1./255)
     
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(INPUT_WIDTH, INPUT_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    return val_generator

def add_new_last_layer(base_model, num_classes):
    model = models.Sequential()
    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model = Model(inputs = base_model.input, outputs = model(base_model.output))
    return model

def train_model(base_model, model, train_generator, val_generator, batch_size):
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=optimizers.RMSprop(lr=2e-3, decay=0.5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    filepath = "checkpoint-"+ model_type+"-"+optimizer_name+"-epoch{epoch:02d}-val_acc{val_acc:.3f}.hdf5"
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=5, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    
    callbacks_list = [checkpoint, early_stop]
 
    history = model.fit_generator(train_generator,
                    epochs=num_epochs,
                    validation_data=val_generator,
                    steps_per_epoch = nTrain // batch_size,
                    validation_steps = nVal // batch_size,
                    callbacks = callbacks_list)
    
    # History: a record of training loss values and metrics values at successive epochs,
    # as well as validation loss values and validation metrics values (if applicable).
    # print(history)
    return model, history

def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()

  return

train_dir, val_dir, nTrain, nVal = split_data(root_dir, 0.8)
conv_base = load_model()
train_generator = extract_features_train(conv_base, train_dir, nTrain, batch_size, num_classes)
val_generator = extract_features_val(conv_base, val_dir, nVal, batch_size, num_classes)
model = add_new_last_layer(conv_base, num_classes)
model, history = train_model(conv_base, model, train_generator, val_generator, batch_size)
plot_training(history)
