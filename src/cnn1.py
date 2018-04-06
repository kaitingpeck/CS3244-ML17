import os, os.path
import glob
import shutil
from keras.applications import VGG16, VGG19, InceptionV3
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
import pandas
from pandas import DataFrame
import traceback

## Set parameters
# Run index (for recording purposes)
run_idx = 1

# File paths
root_dir = 'C:/Users/Kai/Desktop/CS3244/Project/data/test-runs-cnn-scratch/' + str(run_idx)
src_dir_all_data = 'C:/Users/Kai/Desktop/CS3244/Project/data/dataset-resized/full-data'
label_filepath = 'C:/Users/Kai/Desktop/CS3244/Project/data/labels/zero-indexed-files.txt'

# Training paramaters
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
NUM_CLASSES = len(CLASSES)
batch_size = 20
num_epochs = 50
MODEL_TYPE = "CNN"
OPTIMIZER_NAME = "RMSPROP"
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_DEPTH = 3 #RGB

def run_k_fold(label_filepath, root_dir, img_src_dir):
    df = pandas.read_csv(label_filepath, sep = " ", header = None, names = ['imageName','label']) 
    kf = KFold(n_splits=5, shuffle=True)
    imageNames = df['imageName'].values.tolist()
    labels = df['label'].values.tolist()
    
    i = 1
    for train_idx, val_idx in kf.split(imageNames, labels):
        # start a new directory for this fold
        n_dir = make_n_fold_dir(root_dir, i)
        train_dir, val_dir = make_train_val_dir(n_dir)

        # obtain training image (names) and class labels (0,...,6) using the indices
       
        train_imgs, val_imgs = [imageNames[j] for j in train_idx], [imageNames[j] for j in val_idx]
        train_labels, val_labels = [labels[j] for j in train_idx], [labels[j] for j in val_idx] 

        nTrain = len(train_labels)
        nVal = len(val_labels)
        
        copy_images_into_dir(train_imgs, val_imgs, train_labels, val_labels, train_dir, val_dir, img_src_dir)
        try:
            train_model(batch_size, train_dir, val_dir, nTrain, nVal, i)
        except Exception as e:
            print(traceback.format_exc())
            pass

        i += 1
    return

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def count_files(dir_name):
    files = next(os.walk(dir_name))[2]
    num_files = len(files)
    return num_files

def make_n_fold_dir(dst_dir, n_fold):
    n_dir = make_dir(dst_dir + '/' + 'fold-' + str(n_fold))
    return n_dir

def make_train_val_dir(dst_dir):
    train_dir = make_dir(dst_dir + '/train')
    val_dir = make_dir(dst_dir + '/val')

    for item in CLASSES:
        make_dir(train_dir + '/' + item)
        make_dir(val_dir + '/' + item)
        
    return train_dir, val_dir

def copy_images_into_dir(train_imgs, val_imgs, train_labels, val_labels, train_dir, val_dir, src_dir):
    for i in range(len(train_imgs)):
        shutil.copy2(str(src_dir + '\\' + train_imgs[i]), str(train_dir + '/' + CLASSES[train_labels[i]]))
    for i in range(len(val_imgs)):
        shutil.copy2(str(src_dir + '\\' + val_imgs[i]), str(val_dir + '/' + CLASSES[val_labels[i]]))
    return

# Load pre-trained model
def make_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), input_shape=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model

def compile_model(model):    
    model.compile(optimizer=optimizers.rmsprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def make_train_data_generator(train_dir, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
     
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(INPUT_WIDTH, INPUT_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    return train_generator

def make_val_data_generator(val_dir, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
     
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(INPUT_WIDTH, INPUT_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    return val_generator

# Plot results for n-th fold
def plot_results(history, n): 
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.figure()
  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')
  plt.savefig('accuracy' + '-' + str(n) + '-fold.jpg', bbox_inches='tight')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  fig = plt.title('Training and validation loss')
  plt.savefig('loss' + '-' + str(n) + '-fold.jpg', bbox_inches='tight')
  return

def train_model(batch_size, train_dir, val_dir, nTrain, nVal, n):

    # Set up generators to flow batches of data
    train_generator = make_train_data_generator(train_dir, batch_size)
    val_generator = make_val_data_generator(val_dir, batch_size)

    # Set up model
    model = make_model()
    model = compile_model(model)

    # Set parameters for running model
    filepath = "checkpoint-"+ MODEL_TYPE +"-" + OPTIMIZER_NAME + "-epoch{epoch:02d}-val_acc{val_acc:.3f}.hdf5"
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint, early_stop]

    # Run model
    history = model.fit_generator(train_generator,
                    epochs=num_epochs,
                    validation_data=val_generator,
                    steps_per_epoch = nTrain // batch_size,
                    validation_steps = nVal // batch_size,
                    callbacks = callbacks_list, verbose=2)
    
    # History: a record of training loss values and metrics values at successive epochs,
    # as well as validation loss values and validation metrics values (if applicable).

    # Plot results based on this run
    plot_results(history,n)
 
    return

# Create new directory for this run as required
make_dir(root_dir)

# run k-fold cross validation
#label_filepath = .txt file consisting of all samples and labels;
#root_dir = location to store all the copied data for training and validation
run_k_fold(label_filepath, root_dir, src_dir_all_data)
