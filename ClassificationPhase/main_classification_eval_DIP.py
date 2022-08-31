###################  Classifier Evaluation for the Classification Phase ##################
## This script performs the evaluation of the classification of the patch images from   ##
## the CBIS-DDSM dataset after the imputation by the DIP algorithm.                     ##
##########################################################################################

import tensorflow.compat.v1 as tfv1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tfv1.logging.set_verbosity(tfv1.logging.ERROR)
#tfv1.disable_v2_behavior()  # Only needed with tensorflow v2...

import utils.image as image_utils
import utils.reproducibility as rep_utils
import dip_code as dip

from skimage.metrics import structural_similarity as ssim

import time
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support

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
IMAGES_SUB_CLASS = ["Mass", "Calc"]

# ----- CONFIGURATION -----
NUMBER_RUNS = 30  # Number of executions for each configuration.
MISSING_RATES =  [5,10,20,30,60,80]  #Values between 0 and 100.
DATA_SPLIT = [0.7, 0, 0.3]  # Order: Train, Validation, Test. Values between 0 and 1.
INPUTE_NAN = False  # True places missing data as nan, False pre-imputes with 0.
SOURCE = "CBIS_DDSM_100"  
MODEL = 'DIP'  # Accepted values: KNN, MICE.
START_RUN = 0  # Helpful when resuming an experiment...
NUMBER_IMAGES_TO_SAVE = 50  # Set to 0 to avoid saving images.  **Try and probably change
RESIZE_DIMENSIONS = (100, 100)  # Size of the saved images.
INVERT = False  # Needed to save images of some datasets.

def mean_absolute_error(x1, x2):
    return np.mean(np.abs(x1 - x2))

def ssim_average(predicted_data,images_test):
    ssim_values = []
    for img_index in range(0, images_test.shape[0]):
      predicted_image = np.reshape(predicted_data[img_index], RESIZE_DIMENSIONS)
      test_image = np.reshape(images_test[img_index], RESIZE_DIMENSIONS)
      ssim_metric = ssim(predicted_image, test_image, data_range=1) 
      ssim_values.append(ssim_metric)
    return (sum(ssim_values) / len(ssim_values))

def eval_by_class(classes, classes_test, images_test, images_with_mv_test, imputed_images,predicted_data, labels_save_test, masks_test, mr, run, label):

    list_classe = np.where(np.array(classes_test) == classes)[0]
    images_test_mb = images_test[list_classe,:,:,:]
    images_with_mv_test_mb = images_with_mv_test[list_classe,:,:,:]
    imputed_images_mb = imputed_images[list_classe,:,:,:]
    predicted_data_mb = predicted_data[list_classe,:,:,:]
    labels_save_test_mb = np.array(labels_save_test)[list_classe]
    masks_test_mb = masks_test[list_classe,:,:]
        
    if NUMBER_IMAGES_TO_SAVE > 0:
        image_utils.save_images(IMAGES_DIR + SOURCE + "_" + MODEL + '_' + IMAGES_SUB_PATH[0] + '_' + label + "/" + str(mr), images_test_mb, SOURCE + "_" + str(run), labels_save_test_mb, NUMBER_IMAGES_TO_SAVE, RESIZE_DIMENSIONS, INVERT)
        image_utils.save_images(IMAGES_DIR + SOURCE + "_" + MODEL + '_' + IMAGES_SUB_PATH[1] + '_' + label + "/" + str(mr), images_with_mv_test_mb, SOURCE + "_" + str(run), labels_save_test_mb, NUMBER_IMAGES_TO_SAVE, RESIZE_DIMENSIONS, INVERT)
        image_utils.save_images(IMAGES_DIR + SOURCE + "_" + MODEL + '_' + IMAGES_SUB_PATH[2] + '_' + label + "/" + str(mr), imputed_images_mb, SOURCE + "_" + str(run), labels_save_test_mb, NUMBER_IMAGES_TO_SAVE, RESIZE_DIMENSIONS, INVERT)
        image_utils.save_images(IMAGES_DIR + SOURCE + "_" + MODEL + '_' + IMAGES_SUB_PATH[3] + '_' + label + "/" + str(mr), predicted_data_mb, SOURCE + "_" + str(run), labels_save_test_mb, NUMBER_IMAGES_TO_SAVE, RESIZE_DIMENSIONS, INVERT)

    mask_test_flat_mb = masks_test_mb.astype(bool).flatten()
    metric_mb = mean_absolute_error(predicted_data_mb.flatten()[mask_test_flat_mb], images_test_mb.flatten()[mask_test_flat_mb])
    ssim_metric = ssim(predicted_data_mb,images_test_mb)
    
    return(metric_mb, ssim_metric)

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
    classes_test_mc = []; classes_test_mb = []  #mc = mass vs cal and mb = malignant vs benign
    for k in range(0,len(labels_save_test)):
        if(labels_save_test[k] in (classes[(classes['Abnormality']=='calcification') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_test_mc.append(1); classes_test_mb.append(1)
        elif(labels_save_test[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_test_mc.append(1); classes_test_mb.append(0)
        elif(labels_save_test[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_test_mc.append(0); classes_test_mb.append(1)
        elif(labels_save_test[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_test_mc.append(0); classes_test_mb.append(0)
    classes_train_mc = []; classes_train_mb = []
    for k in range(0,len(labels_save_train)):
        if(labels_save_train[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_train_mc.append(1); classes_train_mb.append(1)
        elif(labels_save_train[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_train_mc.append(1); classes_train_mb.append(0)
        elif(labels_save_train[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_train_mc.append(0); classes_train_mb.append(1)
        elif(labels_save_train[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_train_mc.append(0); classes_train_mb.append(0)
    classes_val_mc = []; classes_val_mb = []
    for k in range(0,len(labels_save_val)):
        if(labels_save_val[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_val_mc.append(1); classes_val_mb.append(1)
        elif(labels_save_val[k] in (classes[ (classes['Abnormality']=='calcification') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_val_mc.append(1); classes_val_mb.append(0)
        elif(labels_save_val[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='BENIGN')])['Name'].values):
            classes_val_mc.append(0); classes_val_mb.append(1)
        elif(labels_save_val[k] in (classes[ (classes['Abnormality']=='mass') & (classes['Pathology']=='MALIGNANT')])['Name'].values):
            classes_val_mc.append(0); classes_val_mb.append(0)
    classes_train_mc = np.array(classes_train_mc); classes_test_mc = np.array(classes_test_mc); classes_val_mc = np.array(classes_val_mc)
    classes_train_mb = np.array(classes_train_mb); classes_test_mb = np.array(classes_test_mb); classes_val_mb = np.array(classes_val_mb)
    
    del test, train, mass_ben_train, mass_mal_train, calc_ben_train, calc_mal_train, labels_save 
    del images
        
    return(images_test, classes_test_mc, classes_test_mb , labels_save_test)  


def main():
    print("Images are being processed...")
    images,label = image_utils.load_dataset_patch(SOURCE)
    classes = pd.read_csv('CBIS_DDSM_images_labels.csv')

    for run in range(0, NUMBER_RUNS):
        for mr in MISSING_RATES:

            print("Configuration: " + str(mr) + " / " + str(run + 1) + "...")
    
            images_test, classes_test_mc, classes_test_mb , labels_save_test = pre_datasets(run,mr,images,label, classes)


            masks_neg = rep_utils.get_missing_masks(REPRODUCIBILITY_DIR, SOURCE, (images_test.shape[0], images_test.shape[1], images_test.shape[2]), mr / 100)
            if images_test.shape[3] == 1:
                masks_neg = np.reshape(masks_neg, [-1, images_test.shape[1], images_test.shape[2], images_test.shape[3]])
            else:
                masks_neg = np.stack((masks_neg,) * 3, axis=-1)

            masks_test = (~masks_neg.astype(bool)).astype(int)
            if INPUTE_NAN:
                images_with_mv_test = images_test * masks_neg + -1 * masks_test
                images_with_mv_test[images_with_mv_test == -1] = np.nan
            else:
                constant = 0.0
                images_with_mv_test = images_test * masks_neg + constant * masks_test
        
            del masks_neg


            images = []
            for k in range(0,images_with_mv_test.shape[0]):
                if(k% 20 == 0):
                    print("DIP: Imputing Image " + str(k) + " of " + str(images_test.shape[0]))
                img_np = np.reshape(images_test[k],(1,images_test.shape[1],images_test.shape[2]))
                img_mask_np = abs(np.reshape(masks_test[k],(1,masks_test.shape[1],masks_test.shape[2]))-1)
                out_np = dip.training_cycle(img_np,img_mask_np)
                out_np = np.reshape(out_np,(masks_test.shape[1],masks_test.shape[2]))
                images.append(out_np)
            
            predicted_data = np.expand_dims(np.asarray(images), axis=3)   

            if(INPUTE_NAN):
            	imputed_images = predicted_data
            else:
            	imputed_images = images_with_mv_test * (~masks_test.astype(bool)).astype(int) + predicted_data * masks_test


            ###Separate by classes Mass and Calc
    
            metric_mass, ssim_mass, psnr_mass = eval_by_class(0, classes_test_mc, images_test, images_with_mv_test, imputed_images, predicted_data, labels_save_test, masks_test, mr, run, 'Mass')
            metric_calc, ssim_calc, psnr_calc = eval_by_class(1, classes_test_mc, images_test, images_with_mv_test, imputed_images, predicted_data, labels_save_test, masks_test, mr, run,'Calc') 

            model = keras.models.load_model("CNN_mass_calc_CBIS_DDSM/model_cnn_2cl_best_"+str(run)+".h5")
            predict = model.predict(imputed_images,batch_size =1)
            y_classes = 1*(predict>0.5)
            accuracy_mc = accuracy_score(classes_test_mc, y_classes)
            print(confusion_matrix(classes_test_mc, y_classes, labels =[0,1]))  
            datas = precision_recall_fscore_support(classes_test_mc, y_classes, average=None, labels=[0, 1])
            print(datas)
            precision_mass, precision_calc = datas[0]
            recall_mass, recall_calc = datas[1]
            fscore_mass, fscore_calc = datas[2]
            print(classification_report(classes_test_mc, y_classes, digits=4 , target_names=['mass', 'calc']))
    
            ###Separate by classes Malignant and Benign
            metric_mal, ssim_mal, psnr_mal = eval_by_class(0, classes_test_mb, images_test, images_with_mv_test, imputed_images, predicted_data, labels_save_test, masks_test, mr, run, 'Malignant')
            metric_ben, ssim_ben, psnr_ben = eval_by_class(1, classes_test_mb, images_test, images_with_mv_test, imputed_images, predicted_data, labels_save_test, masks_test, mr, run,'Benign') #alterar para 2 quando for o caso das 4 classes

            model = keras.models.load_model("CNN_mal_ben_CBIS_DDSM/model_cnn_2cl_best_"+str(run)+".h5")
            predict = model.predict(imputed_images,batch_size =1)
            y_classes = 1*(predict>0.5)
            accuracy_mb = accuracy_score(classes_test_mb, y_classes)
            print(confusion_matrix(classes_test_mb, y_classes, labels =[0,1]))  
            datas = precision_recall_fscore_support(classes_test_mb, y_classes, average=None, labels=[0, 1])
            print(datas)
            precision_mal, precision_ben = datas[0]
            recall_mal, recall_ben = datas[1]
            fscore_mal, fscore_ben = datas[2]
            print(classification_report(classes_test_mb, y_classes, digits=4 , target_names=['mass', 'calc']))

            rep_utils.write_line_to_csv(
                RESULTS_BASE_PATH, SOURCE + "_" + MODEL + "_2classes" + ".csv",
                {
                    "MR": mr,
                    "RUN": (run + 1),
                    MODEL + "_MAE_mass": metric_mass,
                    MODEL + "_SSIM_mass": ssim_mass,
                    MODEL + "_PSNR_mass": psnr_mass,
                    MODEL + "_MAE_calc": metric_calc,
                    MODEL + "_SSIM_calc": ssim_calc,
                    MODEL + "_PSNR_calc": psnr_calc,
                    MODEL + "_MAE_mal": metric_mal,
                    MODEL + "_SSIM_mal": ssim_mal,
                    MODEL + "_PSNR_mal": psnr_mal,
                    MODEL + "_MAE_ben": metric_ben,
                    MODEL + "_SSIM_ben": ssim_ben,
                    MODEL + "_PSNR_ben": psnr_ben,
                    "Accuracy_mass_calc": accuracy_mc,
                    "Precision_mass": precision_mass,
                    "Recall_mass": recall_mass,
                    "Fscore_mass": fscore_mass,
                    "Precision_calc": precision_calc,
                    "Recall_calc": recall_calc,
                    "Fscore_calc": fscore_calc,
                    "Accuracy_mal_ben": accuracy_mb,
                    "Precision_mal": precision_mal,
                    "Recall_mal": recall_mal,
                    "Fscore_mal": fscore_mal,
                    "Precision_ben": precision_ben,
                    "Recall_ben": recall_ben,
                    "Fscore_ben": fscore_ben
                })


if __name__ == '__main__': 

    start_time = time.time()
    main()
    print("\nExecution time: %s seconds\n" % (time.time() - start_time))