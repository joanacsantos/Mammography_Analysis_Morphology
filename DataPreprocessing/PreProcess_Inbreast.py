###########################  Preprocessing of INbreast dataset ###########################
## This script reads the DICOM files from the dataset and saves the final version of    ##
## the images into a png format for latter use.                                         ##
## The dataset is publicly available at :                                               ##
## http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database  ##
##########################################################################################

from PIL import Image, ImageOps
import pydicom
import glob
import os
import numpy as np
import cv2 as cv2
import os

# Setting the dataset directory
thisdir = "*****"

images = []; labels = []
tt = 0

for img_path in sorted(glob.glob(thisdir + "\*.dcm")):
    ds = pydicom.dcmread(img_path)
    
    image = ds.pixel_array
    
    dir1 = img_path.split("\\")
    dir2 = dir1[1]
    dir3 = dir2[:-4]
    
    ret,thresh1 = cv2.threshold(np.array(image), 20, 255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels_im = cv2.connectedComponents(opening)
    values = []
    for value in range(1,num_labels):
        values.append(len(np.where(labels_im==value)[0]))
        
    final_mask = labels_im == np.where(values == np.amax(values))[0][0] +1
    
    new_image = img*final_mask
    
    image = cv2.resize(new_image,(256,256), interpolation = cv2.INTER_AREA)
    image = (image - np.min(image))/np.ptp(image)
    
    tt = tt +1
    if(tt%100 == 0):
        print('Saving image: {} of 410'.format(tt))
    img = Image.fromarray(np.uint8(image*255))
    img.save('INbreast_256/' + dir3 + '.png')
