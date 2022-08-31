###########################  Base Code for the Imputation Phase ##########################
## This script performs the imputation for the VAE, KNN, MICE and MC algorithm.         ##
## This script is applicable on full images from the MIAS, INbreast and CSAW datasets.  ##
##########################################################################################

import tensorflow.compat.v1 as tfv1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tfv1.logging.set_verbosity(tfv1.logging.ERROR)
tfv1.disable_v2_behavior()  # Only needed with tensorflow v2...
import time
import utils.image as image_utils
import utils.reproducibility as rep_utils
import numpy as np
from sklearn.model_selection import train_test_split
from algorithms.imputer import VAEWrapper, KNNWrapper, MICEWrapper, MCWrapper
from skimage.metrics import structural_similarity as ssim

# -------- SOURCE ---------
REPRODUCIBILITY_DIR = "output/reproducibility/"
RESULTS_BASE_PATH = "output/results/"
IMAGES_DIR = "output/images/"
IMAGES_SUB_PATH = ["Original", "MissingValues", "Imputed"]
MODELS_DIR = "output/models/"
MODEL_TO_CLASS = {'VAE': VAEWrapper,'KNN': KNNWrapper, 'MICE': MICEWrapper, 'MC': MCWrapper}

# ----- CONFIGURATION -----
NUMBER_RUNS = 30  # Number of executions for each configuration.
MISSING_RATES =  [5,10,20,30,60,80]  # Values between 0 and 100.
DATA_SPLIT = [0.7, 0, 0.3]  # Order: Train, Validation, Test. Values between 0 and 1.
INPUTE_NAN = True  # True places missing data as nan, False pre-imputes with 0.
SOURCE = "INbreast_256"  # For the 3 datasets: MIAS_256, INbreast_256 and CSAW_256
MODEL = 'KNN'  # Accepted values: VAE, KNN, MICE, MC.
TRAINING = False #If the algorithm requires training
START_RUN = 0  # Helpful when resuming an experiment...
NUMBER_IMAGES_TO_SAVE = 50  # Set to 0 to avoid saving images.
SAVE_RESULTS = True #If you want to save the complete results
RESIZE_DIMENSIONS = (256, 256)  # Size of the saved images.
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

def single_run_process(run, mr, np_images,label,mask):

    order = rep_utils.get_run_shuffle(REPRODUCIBILITY_DIR, SOURCE, run)
    images = np_images.copy()[order]
    labels_save = [label[i] for i in order]
    mask_r = mask.copy()[order]
            
    masks_neg = rep_utils.get_missing_masks(REPRODUCIBILITY_DIR, SOURCE, (images.shape[0], images.shape[1], images.shape[2]), mr / 100)
    masks_neg = np.expand_dims(masks_neg, axis=3).astype(int)    
    if images.shape[3] == 1:
        masks_neg = np.reshape(masks_neg, [-1, images.shape[1], images.shape[2], images.shape[3]])
    else:
        masks_neg = np.stack((masks_neg,) * 3, axis=-1)

    masks = (~masks_neg.astype(bool)).astype(int)
    masks = masks * np.expand_dims(mask_r, axis=3).astype(int)

    if INPUTE_NAN:
        images_with_mv = images * masks_neg + -1 * masks
        images_with_mv[images_with_mv == -1] = np.nan
    else:
        constant = 0.0
        images_with_mv = images * masks_neg + constant * masks
        
    del masks_neg
    
    train, test = train_test_split(order, test_size=DATA_SPLIT[2], shuffle=False)

    images_train_val = images[train]; images_test = images[test]
    labels_save_test = [labels_save[i] for i in test]
    images_with_mv_train_val = images_with_mv[train]; images_with_mv_test = images_with_mv[test]
    masks_train_val = masks[train]; masks_test = masks[test]
    
    del test, train, labels_save 
    del images, images_with_mv, masks

    begin = time.time()
    
    model = MODEL_TO_CLASS[MODEL](images_train_val, images_test, images_with_mv_train_val, images_with_mv_test, masks_train_val, masks_test, DATA_SPLIT)
    del images_train_val, images_with_mv_train_val, masks_train_val    
    
    if(TRAINING):
        model.train()
        
    predicted_data = model.predict(images_with_mv_test, masks_test)   
    if(INPUTE_NAN):
        imputed_images = predicted_data
    else:
        imputed_images = images_with_mv_test * (~masks_test.astype(bool)).astype(int) + predicted_data * masks_test

    time_value = (time.time()-begin)/60

    if NUMBER_IMAGES_TO_SAVE > 0:
        image_utils.save_images(IMAGES_DIR + SOURCE + "_" + MODEL + "_" + IMAGES_SUB_PATH[0] + "/" + str(mr), images_test, SOURCE + "_" + str(run), labels_save_test, NUMBER_IMAGES_TO_SAVE, RESIZE_DIMENSIONS, INVERT)
        image_utils.save_images(IMAGES_DIR + SOURCE + "_" + MODEL + "_" + IMAGES_SUB_PATH[1] + "/" + str(mr), images_with_mv_test, SOURCE + "_" + str(run), labels_save_test, NUMBER_IMAGES_TO_SAVE, RESIZE_DIMENSIONS, INVERT)
        image_utils.save_images(IMAGES_DIR + SOURCE + "_" + MODEL + "_" + IMAGES_SUB_PATH[2] + "/" + str(mr), imputed_images, SOURCE + "_" + str(run), labels_save_test, NUMBER_IMAGES_TO_SAVE, RESIZE_DIMENSIONS, INVERT)

    if SAVE_RESULTS:
        if not os.path.exists(IMAGES_DIR):
            os.makedirs(IMAGES_DIR)
        np.save(IMAGES_DIR + SOURCE + "_" + MODEL + "_" + str(mr)+ "_" + str(run)+".npy", imputed_images)

    mask_test_flat = masks_test.astype(bool).flatten()
    metric = mean_absolute_error(predicted_data.flatten()[mask_test_flat], images_test.flatten()[mask_test_flat])
    ssim_metric = ssim_average(predicted_data,images_test)

    print(MODEL + " - Run " + str(run + 1) + " - MAE: " + str(metric) + " - SSIM: " + str(ssim_metric))

    return(metric,ssim_metric,time_value)

def main():

    print("Images are being processed...")
    images,label,mask = image_utils.load_dataset_full(SOURCE)

    for mr in MISSING_RATES:
        for run in range(0, NUMBER_RUNS):
            print("Configuration: " + str(mr) + " / " + str(run + 1) + "...")

            metric,ssim_v,time_value = single_run_process(run,mr,images,label,mask)

            rep_utils.write_line_to_csv(
                RESULTS_BASE_PATH, SOURCE + "_" + MODEL + ".csv",
                {
                    "MR": mr,
                    "RUN": (run + 1),
                    MODEL + "_time": time_value,
                    MODEL + "_MAE": metric,
                    MODEL + "_SSIM": ssim_v
                })
    
def main_reproducibility():

    if os.path.exists(REPRODUCIBILITY_DIR + SOURCE):
        return

    print("Creating a reproducible environment for source '" + SOURCE + "'...")
    images,_,_ = image_utils.load_dataset_full(SOURCE)
    rep_utils.save_runs_shuffle(REPRODUCIBILITY_DIR, SOURCE, NUMBER_RUNS, images.shape[0])

if __name__ == '__main__': 

    start_time = time.time()
    main_reproducibility()
    main()
    print("\nExecution time: %s seconds\n" % (time.time() - start_time))