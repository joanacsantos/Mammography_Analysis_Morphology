########################### Preprocessing of CBIS-DDSM dataset ###########################
## This script reads the DICOM files from the dataset and saves the final version of    ##
## the images into a png format for latter use.                                         ##
## The dataset is publicly available at :                                               ##
## https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM                       ##
##########################################################################################

import os
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import matplotlib
import cv2

# Setting the dataset directory
thisdir = "*****"

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(thisdir):
    for file in f:
        if file.endswith(".dcm"):
            files.append(os.path.join(r, file))
  
tt=0
sizes = []
dim = []
for k in files:
    dataset = pydicom.dcmread(k)
    d = dataset.pixel_array
    resized = cv2.resize(d, (100,100), interpolation = cv2.INTER_AREA)
    normalized_d = ((resized - np.amin(resized))/(np.amax(resized) - np.amin(resized)) * 255)

    tt = tt +1
    if(tt%100 == 0):
        print('Saving image: {} of 3568'.format(tt))
    matplotlib.image.imsave('CBIS_DDSM_100/' + k.split("\\")[6] + '.png', normalized_d,cmap='gray')
    