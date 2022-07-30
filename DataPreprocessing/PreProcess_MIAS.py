########################## Preprocessing of Mini MIAS dataset ############################
## This script reads the DICOM files from the dataset and saves the final version of    ##
## the images into a png format for latter use.                                         ##
## The dataset is publicly available at :                                               ##
## http://peipa.essex.ac.uk/info/mias.html                                              ##
##########################################################################################

from PIL import Image, ImageOps
import glob
import os
import numpy as np
import cv2 as cv2

# Setting the dataset directory
thisdir = "*****"

images = []; labels = []
Tt = 0

for img_path in sorted(glob.glob(thisdir + "\*.pgm")):
    img = Image.open(img_path)
    ret,thresh1 = cv2.threshold(np.array(img), 20, 255,cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels_im = cv2.connectedComponents(opening)
    
    values = []
    for value in range(1,num_labels):
        values.append(len(np.where(labels_im==value)[0]))
    
    final_mask= labels_im == np.where(values == np.amax(values))[0][0] +1
    
    dir1 = img_path.split(".")
    dir2 = dir1[0]
        
    img_noback = img*final_mask
    
    img_final = cv.resize(img_noback,(256,256), interpolation = cv.INTER_AREA)
    tt = tt +1
    if(tt%100 == 0):
        print('Saving image: {} of 322'.format(tt))
    img_final = Image.fromarray(np.uint8(img_final))
    img_final.save('MIAS_256/' + dir2 + '.png')
    