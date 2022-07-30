###########################  Preprocessing of CSAW dataset ###########################
## This script reads the DICOM files from the dataset and saves the final version of    ##
## the images into a png format for latter use.                                         ##
## The dataset must be requested to the authors of the paper.                           ##
##########################################################################################

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pydicom
import matplotlib
import cv2


# Setting the dataset directory
thisdir = "*****"

data = pd.read_csv('anon_dataset_nonhidden_211125.csv',delimiter=',')
files = []

#--------Select Health Patients----------------
list_healthy = data.loc[data['x_case']==0]

healthy_left = list_healthy.loc[(list_healthy['viewposition']=='CC') & (list_healthy['imagelaterality']=='Left')]
n = random.sample(list(healthy_left.index),63)
for i in n:
    name_CC = healthy_left['anon_filename'][i]
    files.append(name_CC)
    name_MLO = name_CC.split('_')
    name_MLO[3] = 'MLO'
    name_MLO = '_'.join(name_MLO)
    files.append(name_MLO)

healthy_right = list_healthy.loc[(list_healthy['viewposition']=='CC') & (list_healthy['imagelaterality']=='Right')]
n = random.sample(list(healthy_right.index),63)
for i in n:
    name_CC = healthy_right['anon_filename'][i]
    files.append(name_CC)
    name_MLO = name_CC.split('_')
    name_MLO[3] = 'MLO'
    name_MLO = '_'.join(name_MLO)
    files.append(name_MLO)

#--------Select In Situ Patients----------------
list_cancer1 = data.loc[data['x_type']==1]
cancer1_left = list_cancer1.loc[(list_cancer1['viewposition']=='CC') & (list_cancer1['imagelaterality']=='Left')]
n = random.sample(list(cancer1_left.index),63)
for i in n:
    name_CC = cancer1_left['anon_filename'][i]
    files.append(name_CC)
    name_MLO = name_CC.split('_')
    name_MLO[3] = 'MLO'
    name_MLO = '_'.join(name_MLO)
    files.append(name_MLO)

cancer1_right = list_cancer1.loc[(list_cancer1['viewposition']=='CC') & (list_cancer1['imagelaterality']=='Right')]
n = random.sample(list(cancer1_right.index),63)
for i in n:
    name_CC = cancer1_right['anon_filename'][i]
    files.append(name_CC)
    name_MLO = name_CC.split('_')
    name_MLO[3] = 'MLO'
    name_MLO = '_'.join(name_MLO)
    files.append(name_MLO)

#--------Select Invasive 1 Patients----------------
list_cancer2 = data.loc[data['x_type']==2]
cancer2_left = list_cancer2.loc[(list_cancer2['viewposition']=='CC') & (list_cancer2['imagelaterality']=='Left')]
n = random.sample(list(cancer2_left.index),62)
for i in n:
    name_CC = cancer2_left['anon_filename'][i]
    files.append(name_CC)
    name_MLO = name_CC.split('_')
    name_MLO[3] = 'MLO'
    name_MLO = '_'.join(name_MLO)
    files.append(name_MLO)

cancer2_right = list_cancer2.loc[(list_cancer2['viewposition']=='CC') & (list_cancer2['imagelaterality']=='Right')]
n = random.sample(list(cancer2_right.index),62)
for i in n:
    name_CC = cancer2_right['anon_filename'][i]
    files.append(name_CC)
    name_MLO = name_CC.split('_')
    name_MLO[3] = 'MLO'
    name_MLO = '_'.join(name_MLO)
    files.append(name_MLO)

#--------Select Invasive 2 Patients----------------
list_cancer3 = data.loc[data['x_type']==3]
cancer3_left = list_cancer3.loc[(list_cancer3['viewposition']=='CC') & (list_cancer3['imagelaterality']=='Left')]
n = random.sample(list(cancer3_left.index),62)
for i in n:
    name_CC = cancer3_left['anon_filename'][i]
    files.append(name_CC)
    name_MLO = name_CC.split('_')
    name_MLO[3] = 'MLO'
    name_MLO = '_'.join(name_MLO)
    files.append(name_MLO)

cancer3_right = list_cancer3.loc[(list_cancer3['viewposition']=='CC') & (list_cancer3['imagelaterality']=='Right')]
n = random.sample(list(cancer3_right.index),62)
for i in n:
    name_CC = cancer3_right['anon_filename'][i]
    files.append(name_CC)
    name_MLO = name_CC.split('_')
    name_MLO[3] = 'MLO'
    name_MLO = '_'.join(name_MLO)
    files.append(name_MLO)

n_images = 2000
labels = []
images_final = np.zeros([n_images,256,256])

tt=0
for k in range(0, n_images):
    dataset = pydicom.dcmread(thisdir + '/'+files[k])
    d = dataset.pixel_array
    resized = cv2.resize(d, (256,256), interpolation = cv2.INTER_AREA)
    normalized_d = ((resized - np.amin(resized))/(np.amax(resized) - np.amin(resized)) * 255)
    
    normalized_d = np.array(normalized_d, dtype=np.uint8)
    ret,thresh1 = cv2.threshold(normalized_d, 3, 255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels_im = cv2.connectedComponents(opening)
    values = []
    for value in range(1,num_labels):
        values.append(len(np.where(labels_im==value)[0]))
        
    final_mask = labels_im == np.where(values == np.amax(values))[0][0] +1
    
    img_noback = normalized_d*final_mask
                
    normalized_image = img_noback/255
    
    tt = tt +1
    if(tt%100 == 0):
        print('Saving image: {} of 2000'.format(tt))
    matplotlib.image.imsave('CSAW_256/' + files[k].split("/")[5].split(".")[0] + '.png', img_noback,cmap='gray')
