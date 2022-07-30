"""General-purpose test script for image-to-image translation.
Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.
    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import util, html

import numpy as np
import ntpath
import util.reproducibility as rep_utils
from skimage.metrics import structural_similarity as ssim
import util.image as image_utils

#Setting the basis for this code
# -------- SOURCE ---------
REPRODUCIBILITY_DIR = "output/reproducibility/"
RESULTS_BASE_PATH = "output/results/"
IMAGES_DIR = "output/images/"
IMAGES_SUB_PATH = ["Original", "MissingValues", "Imputed"]
MODELS_DIR = "output/models/"

# ----- CONFIGURATION -----
MODEL = 'CycleGAN'  
NUMBER_IMAGES_TO_SAVE = 50  # Set to 0 to avoid saving images.
SAVE_RESULTS = True #If you want to save the complete results
INVERT = False  # Needed to save images of some datasets.

def mean_absolute_error(x1, x2):
    return np.mean(np.abs(x1 - x2))

def ssim_average(predicted_data,images_test):
    ssim_values = []
    for img_index in range(0, images_test.shape[0]):
      predicted_image = np.reshape(predicted_data[img_index], (images_test.shape[1],images_test.shape[1]))
      test_image = np.reshape(images_test[img_index], (images_test.shape[1],images_test.shape[1]))
      ssim_metric = ssim(predicted_image, test_image, data_range=1) 
      ssim_values.append(ssim_metric)
    return (sum(ssim_values) / len(ssim_values))

def main_test(run,m_rate,dim_image,source):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.dataroot = 'datasets/testA'
    opt.no_dropout = True
    opt.load_size = dim_image
    opt.crop_size = dim_image

    opt.run = run
    opt.m_rate = m_rate
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    images_missing = np.zeros(shape=(len(dataset), dim_image, dim_image,1))
    labels_missing = []
    images_fake = np.zeros(shape=(len(dataset), dim_image, dim_image,1))
    counter = 0
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        short_path = ntpath.basename(img_path[0])
        image_name = os.path.splitext(short_path)[0]

        #For real original images (dataset A)
        im_real = util.tensor2im(visuals['real']) 
        labels_missing.append(image_name)
        images_missing[counter] = np.dsplit(im_real,im_real.shape[-1])[0] #teve de vir com 3 layers iguais
        if i % 50 == 0:  
            print('processing (%04d)-th image... %s' % (i, img_path))

        #For fake images (dataset B)
        im_fake = util.tensor2im(visuals['fake']) 
        images_fake[counter] = np.dsplit(im_fake,im_fake.shape[-1])[0]
        counter = counter + 1
    images_missing = images_missing/255
    images_fake = images_fake/255

    opt2 = TestOptions().parse()
    opt2.num_threads = 0   
    opt2.batch_size = 1  
    opt2.serial_batches = True 
    opt2.no_flip = True  
    opt2.display_id = -1 
    opt2.dataroot = 'datasets/testB'
    opt2.load_size = dim_image
    opt2.crop_size = dim_image
    dataset_real = create_dataset(opt2)

    
    images_real = np.zeros(shape=(len(dataset_real), dim_image, dim_image,1))
    labels_real = []
    counter = 0
    for i, data in enumerate(dataset_real):
      im_real = util.tensor2im(data['A']) 
      images_real[counter] = np.dsplit(im_real,im_real.shape[-1])[0] #teve de vir com 3 layers iguais
      short_path = ntpath.basename(data['A_paths'][0])
      image_name = os.path.splitext(short_path)[0]
      labels_real.append(image_name)

      counter = counter + 1

    images_real = images_real/255
      
    opt3 = TestOptions().parse()
    opt3.num_threads = 0   
    opt3.batch_size = 1  
    opt3.serial_batches = True 
    opt3.no_flip = True  
    opt3.display_id = -1 
    opt3.dataroot = 'datasets/masks_test'
    opt3.load_size = dim_image
    opt3.crop_size = dim_image
    dataset_masks = create_dataset(opt3)
    
    images_masks = np.zeros(shape=(len(dataset_masks), dim_image, dim_image,1))
    labels_masks = []
    counter = 0
    for i, data in enumerate(dataset_masks):
      im_mask = util.tensor2im(data['A']) 
      images_masks[counter] = np.dsplit(im_mask,im_mask.shape[-1])[0] #teve de vir com 3 layers iguais
      labels_masks.append(ntpath.basename(data['A_paths'][0]))
      counter = counter + 1
    images_masks = images_masks/255
    
    predicted_data = images_missing * (~images_masks.astype(bool)).astype(int) + images_fake * images_masks 

    if NUMBER_IMAGES_TO_SAVE > 0:
        image_utils.save_images(IMAGES_DIR + source + "_" + MODEL + "_" + IMAGES_SUB_PATH[0] + "/" + str(m_rate), images_real, source + "_" + str(run), labels_real, NUMBER_IMAGES_TO_SAVE, (dim_image, dim_image), INVERT)
        image_utils.save_images(IMAGES_DIR + source + "_" + MODEL + "_" + IMAGES_SUB_PATH[1] + "/" + str(m_rate), images_missing, source + "_" + str(run), labels_missing, NUMBER_IMAGES_TO_SAVE, (dim_image, dim_image), INVERT)
        image_utils.save_images(IMAGES_DIR + source + "_" + MODEL + "_" + IMAGES_SUB_PATH[2] + "/" + str(m_rate), predicted_data, source + "_" + str(run), labels_missing, NUMBER_IMAGES_TO_SAVE, (dim_image, dim_image), INVERT)

    if SAVE_RESULTS:
        if not os.path.exists(IMAGES_DIR):
            os.makedirs(IMAGES_DIR)
        np.save(IMAGES_DIR + source + "_" + MODEL + "_" + str(m_rate)+ "_" + str(run)+".npy", predicted_data)

    mask_test_flat = images_masks.astype(bool).flatten()
    metric = mean_absolute_error(predicted_data.flatten()[mask_test_flat], images_real.flatten()[mask_test_flat])
    ssim_metric = ssim_average(predicted_data, images_real)

    print(MODEL + " - Run " + str(run + 1) + " - MAE: " + str(metric) + " - SSIM: " + str(ssim_metric))

