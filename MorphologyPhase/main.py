########################  Base Code for the Morphology Evaluation ########################
## This scrip performs the morphology evaluation for the CBIS-DDSM dataset.             ##
##########################################################################################

import os
import time
import utils.image as image_utils
import utils.reproducibility as rep_utils
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

# -------- SOURCE ---------
REPRODUCIBILITY_DIR = "output/reproducibility/"
RESULTS_BASE_PATH = "output/results/"
IMAGES_DIR = "output/images/"
IMAGES_SUB_PATH = ["Original", "MissingValues", "Imputed"]

# ----- CONFIGURATION -----
NUMBER_RUNS = 30  # Number of executions for each configuration.
MISSING_RATES =  [5,10,20,30,60,80]  # Values between 0 and 100.
DATA_SPLIT = [0.7, 0, 0.3]  # Order: Train, Validation, Test. Values between 0 and 1.
INPUTE_NAN = True  # True places missing data as nan, False pre-imputes with 0.
SOURCE = "CBIS_DDSM_100"  
MODEL = 'KNN'  
START_RUN = 0  # Helpful when resuming an experiment...
RESIZE_DIMENSIONS = (100, 100)  # Size of the saved images.
INVERT = False  # Needed to save images of some datasets.

def avg(array_number):
    return (sum(l) / float(len(l)))

def single_run_process(run, mr, np_images,label): 

    images = np_images.copy()[rep_utils.get_run_shuffle(REPRODUCIBILITY_DIR, SOURCE, run)]
    order = rep_utils.get_run_shuffle(REPRODUCIBILITY_DIR, SOURCE, run)
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

    images_test = images[test]
    labels_save_test = [labels_save[i] for i in test]
    images_with_mv_test = images_with_mv[test]
    masks_test = masks[test]
    
    del test, train, labels_save 
    del images, images_with_mv, masks
    
    imputed_images = np.load(IMAGES_DIR + SOURCE + "_" + MODEL + "_" + str(mr)+ "_" + str(run)+".npy")

      n_pixels_total = []
      n_pixels_reconstructed = []
      n_pixels_reconstructed_erro_1_255 = []
      n_pixels_reconstructed_erro_2_255 = []
      n_pixels_reconstructed_erro_5_255 = []
      n_pixels_reconstructed_erro_5 = []
      n_pixels_reconstructed_erro_10 = []
      n_pixels_reconstructed_erro_15 = []
    

    for img_index in range(0, images_with_mv_test.shape[0]):
      mask_image = np.reshape(masks_test[img_index],(100,100))
      predicted_image = np.reshape(imputed_images[img_index],(100,100))
      test_image = np.reshape(images_test[img_index],(100,100))

      mask_test_flat = mask_image.astype(bool).flatten()
      n_pixels_total.append(sum(mask_test_flat))
      n_pixels_reconstructed.append(sum(abs(predicted_image.flatten()[mask_test_flat]- test_image.flatten()[mask_test_flat]) == 0))
      n_pixels_reconstructed_erro_1_255.append(sum(abs(predicted_image.flatten()[mask_test_flat]- test_image.flatten()[mask_test_flat]) <= 0.004))
      n_pixels_reconstructed_erro_2_255.append(sum(abs(predicted_image.flatten()[mask_test_flat]- test_image.flatten()[mask_test_flat]) <= 0.008))
      n_pixels_reconstructed_erro_5_255.append(sum(abs(predicted_image.flatten()[mask_test_flat]- test_image.flatten()[mask_test_flat]) <= 0.02))
      n_pixels_reconstructed_erro_5.append(sum(abs(predicted_image.flatten()[mask_test_flat]- test_image.flatten()[mask_test_flat]) <= 0.05))
      n_pixels_reconstructed_erro_10.append(sum(abs(predicted_image.flatten()[mask_test_flat]- test_image.flatten()[mask_test_flat]) <= 0.1))
      n_pixels_reconstructed_erro_15.append(sum(abs(predicted_image.flatten()[mask_test_flat]- test_image.flatten()[mask_test_flat]) <= 0.15))

    return(avg(n_pixels_total), avg(n_pixels_reconstructed), avg(n_pixels_reconstructed_erro_1_255), avg(n_pixels_reconstructed_erro_2_255), avg(n_pixels_reconstructed_erro_5_255), avg(n_pixels_reconstructed_erro_5), avg(n_pixels_reconstructed_erro_10), avg(n_pixels_reconstructed_erro_15))

def main():

    print("Images are being processed...")
    images,label = image_utils.load_dataset_patch(SOURCE)

    for mr in MISSING_RATES:
        for run in range(0, NUMBER_RUNS):
            print("Configuration: " + str(mr) + " / " + str(run + 1) + "...")

            n_pixels_total,n_pixels_reconstructed,n_pixels_reconstructed_erro_1_255,n_pixels_reconstructed_erro_2_255,n_pixels_reconstructed_erro_5_255, n_pixels_reconstructed_erro_5, n_pixels_reconstructed_erro_10, n_pixels_reconstructed_erro_15 = single_run_process(run,mr,images,label)

            rep_utils.write_line_to_csv(
                RESULTS_BASE_PATH, SOURCE + "_" + MODEL + "_morphology.csv",
                {
                    "MR": mr,
                    "RUN": (run + 1),
                    MODEL + "_N_Pixels_Missing": n_pixels_total,
                    MODEL + "_N_Pixels_Reconsructed": n_pixels_reconstructed,
                    MODEL + "_N_Pixels_1grayleve": n_pixels_reconstructed_erro_1_255,
                    MODEL + "_N_Pixels_2graylevel": n_pixels_reconstructed_erro_2_255,
                    MODEL + "_N_Pixels_5graylevel": n_pixels_reconstructed_erro_5_255,
                    MODEL + "_N_Pixels_error5": n_pixels_reconstructed_erro_5,
                    MODEL + "_N_Pixels_error10": n_pixels_reconstructed_erro_10,
                    MODEL + "_N_Pixels_error15": n_pixels_reconstructed_erro_15
                })

if __name__ == '__main__': 
    start_time = time.time()
    main()
    print("\nExecution time: %s seconds\n" % (time.time() - start_time))