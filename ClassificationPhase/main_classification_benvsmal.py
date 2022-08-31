######################  Classifier Code for the Classification Phase #####################
## This script performs the classification of benign and malignant patch images from    ##
## the CBIS-DDSM dataset.                                                               ##
##########################################################################################

import tensorflow.compat.v1 as tfv1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tfv1.logging.set_verbosity(tfv1.logging.ERROR)
#tfv1.disable_v2_behavior()  # Only needed with tensorflow v2...

import utils.image as image_utils
import utils.reproducibility as rep_utils

import time
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adagrad
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback

import sys
from keras import optimizers
from keras.layers import Activation
from keras import callbacks
from tensorflow.keras import regularizers

import csv
from pathlib import Path
import random
from scipy.stats import bernoulli

# -------- SOURCE ---------
REPRODUCIBILITY_DIR = "output/reproducibility/"
RESULTS_BASE_PATH = "output/results/"
IMAGES_DIR = "output/images/"
IMAGES_SUB_PATH = ["Original", "MissingValues", "Imputed", "PreImputed"]
MODELS_DIR = "output/models/"

# ----- CONFIGURATION -----
NUMBER_RUNS = 30  # Number of executions for each configuration.
MISSING_RATES =  [5,10,20,30,60,80]  #Values between 0 and 100.
DATA_SPLIT = [0.7, 0, 0.3]  # Order: Train, Validation, Test. Values between 0 and 1.
INPUTE_NAN = True  # True places missing data as nan, False pre-imputes with 0.
SOURCE = "CBIS_DDSM_100"  
TRAINING = False #If the algorithm requires training
START_RUN = 0  # Helpful when resuming an experiment...
NUMBER_IMAGES_TO_SAVE = 50  # Set to 0 to avoid saving images.  **Try and probably change
RESIZE_DIMENSIONS = (100, 100)  # Size of the saved images.
INVERT = False  # Needed to save images of some datasets.

def pre_datasets(run, mr, np_images,label, classes): 
    images = np_images.copy()[rep_utils.get_run_shuffle(REPRODUCIBILITY_DIR, SOURCE, run)]
    
    order = rep_utils.get_run_shuffle(REPRODUCIBILITY_DIR, SOURCE, run)
    labels_save = [label[i] for i in order]
    
    #Divide the dataset in 4 parts: calc and mass + benign and malignant(assure the distribution of the classes)
    calc_ben=[]; calc_mal=[]; mass_ben=[];mass_mal=[]
    for k in range(0,len(labels_save)):
        if(labels_save[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='BENIGN')])['Name'].values):
            calc_ben.append(k)
        elif(labels_save[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            calc_mal.append(k)
        elif(labels_save[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='BENIGN')])['Name'].values):
            mass_ben.append(k)
        elif(labels_save[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            mass_mal.append(k)
            
    calc_ben_train, calc_ben_test = train_test_split(calc_ben, test_size=DATA_SPLIT[2], shuffle=False)
    calc_mal_train, calc_mal_test = train_test_split(calc_mal, test_size=DATA_SPLIT[2], shuffle=False)
    mass_ben_train, mass_ben_test = train_test_split(mass_ben, test_size=DATA_SPLIT[2], shuffle=False)
    mass_mal_train, mass_mal_test = train_test_split(mass_mal, test_size=DATA_SPLIT[2], shuffle=False)
    
    calc_ben_train, calc_ben_val = train_test_split(calc_ben_train, test_size=0.143, shuffle=False)
    calc_mal_train, calc_mal_val = train_test_split(calc_mal_train, test_size=0.143, shuffle=False)
    mass_ben_train, mass_ben_val = train_test_split(mass_ben_train, test_size=0.143, shuffle=False)
    mass_mal_train, mass_mal_val = train_test_split(mass_mal_train, test_size=0.143, shuffle=False)

    del mass_ben, mass_mal, calc_ben, calc_mal, order
    
    train = sorted(mass_ben_train + mass_mal_train + calc_ben_train + calc_mal_train)
    test = sorted(mass_ben_test + mass_mal_test + calc_ben_test + calc_mal_test)
    val = sorted(mass_ben_val + mass_mal_val + calc_ben_val + calc_mal_val)

    images_train = images[train]; images_test = images[test]; images_val = images[val]
    labels_save_test = [labels_save[i] for i in test]; labels_save_train = [labels_save[i] for i in train]; labels_save_val = [labels_save[i] for i in val]

    #The real classes must also be organized in the specific order
    classes_test = []
    for k in range(0,len(labels_save_test)):
        if(labels_save_test[k] in (classes[(classes['Abnormality']=='calcification') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_test.append(1)
        elif(labels_save_test[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_test.append(0)
        elif(labels_save_test[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_test.append(1)
        elif(labels_save_test[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_test.append(0)
    classes_train = []
    for k in range(0,len(labels_save_train)):
        if(labels_save_train[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_train.append(1)
        elif(labels_save_train[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_train.append(0)
        elif(labels_save_train[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_train.append(1)
        elif(labels_save_train[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_train.append(0)
    classes_val = []
    for k in range(0,len(labels_save_val)):
        if(labels_save_val[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_val.append(1)
        elif(labels_save_val[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_val.append(0)
        elif(labels_save_val[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_val.append(1)
        elif(labels_save_val[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_val.append(0)
            
    print('Images for Training: ' + str(len(classes_train)) + '-> ' + str(len(np.where(np.array(classes_train) == 0)[0])) +' malignant images and ' + str(len(np.where(np.array(classes_train) == 1)[0])) + ' benign images')
    print('Images for Testing: ' + str(len(classes_test)) + '-> ' + str(len(np.where(np.array(classes_test) == 0)[0])) +' malignant images and ' + str(len(np.where(np.array(classes_test) == 1)[0])) + ' benign images')
    print('Images for Validation: ' + str(len(classes_val)) + '-> ' + str(len(np.where(np.array(classes_val) == 0)[0])) +' malignant images and ' + str(len(np.where(np.array(classes_val) == 1)[0])) + ' benign images')
    
    del test, train, mass_ben_train, mass_mal_train, calc_ben_train, calc_mal_train, labels_save 
    del images
    
    classes_train = np.array(classes_train); classes_test = np.array(classes_test); classes_val = np.array(classes_val)
    
    return(images_train, images_test, images_val, classes_train, classes_test, classes_val, labels_save_train, labels_save_test, labels_save_val)  


def main():
    print("Images are being processed...")
    images,label = image_utils.load_dataset_patch(SOURCE)
    classes = pd.read_csv('CBIS_DDSM_images_labels.csv')

    for run in range(0, NUMBER_RUNS):
        print("Configuration: " + str(mr) + " / " + str(run + 1) + "...")
    
        images_train, images_test, images_val, classes_train, classes_test, classes_val, labels_save_train, labels_save_test, labels_save_val = pre_datasets(run,mr,images,label, classes)

        """ Parameters """
        img_width, img_height = 100, 100
        batch_size = 32
        classes_num = 2
        lr = 0.001
        epochs = 500

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding ="same", input_shape=(img_width, img_height, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3,3), padding ="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(16))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  #classes_num or 2  

        train_datagen = ImageDataGenerator(rotation_range=180, zoom_range=0.2, shear_range=10, horizontal_flip=True, vertical_flip=True, fill_mode="reflect")
        test_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(images_train, classes_train, batch_size=batch_size)
        validation_generator = train_datagen.flow(images_val, classes_val)
   
        if not os.path.exists('CNN_mal_ben_CBIS_DDSM'):
            os.makedirs('CNN_mal_ben_CBIS_DDSM')

        # Callback for checkpointing
        checkpoint = ModelCheckpoint('CNN_mal_ben_CBIS_DDSM\model_cnn_2cl_best_'+str(run)+'.h5',monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_freq='epoch')
        # Early stopping (stop training after the validation loss reaches the minimum)
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=40, verbose=1)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train
        history_0 = model.fit(
                train_generator,
                steps_per_epoch=len(classes_train) // batch_size,
                epochs=500,
                validation_data=validation_generator,
                callbacks=[checkpoint,earlystopping])
    
        predict = model.predict(images_test,batch_size =1)

        # predict the class label
        y_classes = 1*(predict>0.5)

        print('Accuracy Score: ' + str(accuracy_score(classes_test, y_classes))) 
        print(confusion_matrix(classes_test, y_classes, labels =[0,1]))  
        print('Precision: ' + str(precision_score(classes_test, y_classes, average='micro')))
        print('Recall: ' + str(recall_score(classes_test, y_classes, average='micro')))
        print('F1-score: ' + str(f1_score(classes_test, y_classes, average='micro')))
        print(classification_report(classes_test, y_classes, digits=4))


if __name__ == '__main__': 

    start_time = time.time()
    main()
    print("\nExecution time: %s seconds\n" % (time.time() - start_time))