#########################  CycleGAN Code for the Imputation Phase ########################
## This script reads performs the imputation for the CycleGAN algorithm.                ##
## This script is applicable on full images from the CBIS-DDSM dataset.                 ##
## This code is based on the original paper implementation, available on GitHub.        ##
## https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix                              ##
##########################################################################################

import util.image as image_utils
import util.reproducibility as rep_utils
import time
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import os

import train as train
import test as test

# -------- SOURCE ---------
REPRODUCIBILITY_DIR = "output/reproducibility/"
RESULTS_BASE_PATH = "output/results/"
IMAGES_DIR = "output/images/"
IMAGES_SUB_PATH = ["Original", "MissingValues", "Imputed"]
MODELS_DIR = "output/models/"

# ----- CONFIGURATION -----
NUMBER_RUNS = 30  # Number of executions for each configuration.
MISSING_RATES =  [5,10,20,30,60,80]  # Values between 0 and 100.
DATA_SPLIT = [0.7, 0, 0.3]  # Order: Train, Validation, Test. Values between 0 and 1.
INPUTE_NAN = False  # True places missing data as nan, False pre-imputes with 0.
SOURCE = "INbreast_256"  #######CBIS_DDSM_100
MODEL = 'CycleGAN'  # Accepted values: VAE, KNN, MICE, MC.
TRAINING = True #If the algorithm requires training
START_RUN = 0  # Helpful when resuming an experiment...
NUMBER_IMAGES_TO_SAVE = 50  # Set to 0 to avoid saving images.
SAVE_RESULTS = True #If you want to save the complete results
RESIZE_DIMENSIONS = (100, 100)  # Size of the saved images.
INVERT = False  # Needed to save images of some datasets.

def single_run_process(run, mr, np_images,label): 

    order = rep_utils.get_run_shuffle(REPRODUCIBILITY_DIR, SOURCE, run)
    images = np_images.copy()[order]
    labels_save = [label[i] for i in order]
            
    masks_neg = rep_utils.get_missing_masks(REPRODUCIBILITY_DIR, SOURCE, (images.shape[0], images.shape[1], images.shape[2]), mr / 100)
    if images.shape[3] == 1:
        masks_neg = np.reshape(masks_neg, [-1, images.shape[1], images.shape[2], images.shape[3]])
    else:
        masks_neg = np.stack((masks_neg,) * 3, axis=-1)

    masks = (~masks_neg.astype(bool)).astype(int)
    if INPUTE_NAN:
        images_with_mv = images * masks_neg + -1 * masks
        images_with_mv[images_with_mv == -1] = np.nan
    else:
        constant = 0.0
        images_with_mv = images * masks_neg + constant * masks
        
    del masks_neg
    
    train, test = train_test_split(order, test_size=DATA_SPLIT[2], shuffle=False)

    images_train_val = images[train]; images_test = images[test]
    labels_save_train = [labels_save[i] for i in train]
    labels_save_test = [labels_save[i] for i in test]
    images_with_mv_train_val = images_with_mv[train]; images_with_mv_test = images_with_mv[test]
    masks_train_val = masks[train]; masks_test = masks[test]
    
    del test, train, labels_save 
    del images, images_with_mv, masks
    
    image_utils.save_images("datasets/trainA/", images_with_mv_train_val, SOURCE + "_" + str(run), labels_save_train, None, None,INVERT)
    image_utils.save_images("datasets/trainB/", images_train_val, SOURCE + "_" + str(run), labels_save_train, None, None, INVERT)

    image_utils.save_images("datasets/testA/", images_with_mv_test, SOURCE + "_" + str(run), labels_save_test, None, None, INVERT)
    image_utils.save_images("datasets/testB/", images_test, SOURCE + "_" + str(run), labels_save_test, None, None, INVERT)
    image_utils.save_images("datasets/masks_test/", masks_test, SOURCE + "_" + str(run), labels_save_test, None, None, INVERT)

    return(0)

def main():

    print("Images are being processed...")
    images,label = image_utils.load_dataset_patch(SOURCE)

    for mr in [80]:
        for run in [0]:
            #Remove the previous datasets
            print('Removing previous datasets...')
            shutil.rmtree('checkpoints/mammo100', ignore_errors=True)
            shutil.rmtree('datasets/trainA', ignore_errors=True)
            shutil.rmtree('datasets/trainB', ignore_errors=True)
            shutil.rmtree('datasets/testA', ignore_errors=True)
            shutil.rmtree('datasets/testB', ignore_errors=True)
            shutil.rmtree('datasets/masks_test', ignore_errors=True)
        
            #Create the new datasets for the run
            print('Creating the new datasets...')
            single_run_process(run, mr, images,label)
            seconds = time.time()
            print('Beginning training run '+ str(run+1) + ' for ' + str(mr) + ' missing rate')
            train.main_train(run, mr, RESIZE_DIMENSIONS[0])
            ending = time.time()
            print('Training time (h): ' + str((ending-seconds)/60/60))
        
            #Rename the generator and discriminator for the purposes of transforming B to A
            os.rename('checkpoints/mammo100/latest_net_G_A.pth','checkpoints/mammo100/latest_net_G.pth')
            os.rename('checkpoints/mammo100/latest_net_D_A.pth','checkpoints/mammo100/latest_net_D.pth')  
        
            seconds = time.time()
            print('Beginning testing run '+ str(run+1) + ' for ' + str(mr) + ' missing rate')
            test.main_test(run,mr, RESIZE_DIMENSIONS[0],SOURCE)
            ending = time.time()
            print('Testing time (h): ' + str((ending-seconds)/60/60))

    for mr in MISSING_RATES:
        for run in range(0, NUMBER_RUNS):
            #Remove the previous datasets
            print('Removing previous datasets...')
            shutil.rmtree('checkpoints/mammo100', ignore_errors=True)
            shutil.rmtree('datasets/trainA', ignore_errors=True)
            shutil.rmtree('datasets/trainB', ignore_errors=True)
            shutil.rmtree('datasets/testA', ignore_errors=True)
            shutil.rmtree('datasets/testB', ignore_errors=True)
            shutil.rmtree('datasets/masks_test', ignore_errors=True)
        
            #Create the new datasets for the run
            print('Creating the new datasets...')
            single_run_process(run, mr, images,label)
            seconds = time.time()
            print('Beginning training run '+ str(run+1) + ' for ' + str(mr) + ' missing rate')
            train.main_train(run, mr, RESIZE_DIMENSIONS[0])
            ending = time.time()
            print('Training time (h): ' + str((ending-seconds)/60/60))
        
            #Rename the generator and discriminator for the purposes of transforming B to A
            os.rename('checkpoints/mammo100/latest_net_G_A.pth','checkpoints/mammo100/latest_net_G.pth')
            os.rename('checkpoints/mammo100/latest_net_D_A.pth','checkpoints/mammo100/latest_net_D.pth')  
        
            seconds = time.time()
            print('Beginning testing run '+ str(run+1) + ' for ' + str(mr) + ' missing rate')
            test.main_test(run,mr, RESIZE_DIMENSIONS[0],SOURCE)
            ending = time.time()
            print('Testing time (h): ' + str((ending-seconds)/60/60))
 

def main_reproducibility():

    if os.path.exists(REPRODUCIBILITY_DIR + SOURCE):
        return

    print("Creating a reproducible environment for source '" + SOURCE + "'...")
    images,_ = image_utils.load_dataset_patch(SOURCE)
    rep_utils.save_runs_shuffle(REPRODUCIBILITY_DIR, SOURCE, NUMBER_RUNS, images.shape[0])

if __name__ == '__main__': 
    start_time = time.time()
    main_reproducibility()
    main()
    print("\nExecution time: %s seconds\n" % (time.time() - start_time))