from PIL import Image, ImageOps
import glob
import os
import numpy as np
import cv2
import cv2 as cv2

def read_images_to_np_matrix(folders, resize_dimensions=None, is_grey=True):

    images = []
    min_width = None
    min_height = None

    if resize_dimensions is not None:
        min_width = resize_dimensions[0]
        min_height = resize_dimensions[1]

    for directory in folders:
        for img_path in glob.glob(directory + "/*/*.png"):

            img = Image.open(img_path)
            if is_grey:
                img_float = Image.fromarray(np.divide(np.array(img), 2 ** 8 - 1))
                img = img_float.convert('L')
            images.append(img)

            if resize_dimensions is None:
                width, height = img.size

                if min_width is None or width < min_width:
                    min_width = width

                if min_height is None or height < min_height:
                    min_height = height

    images_np = []

    for img in images:
        img_resized = img.resize((min_width, min_height), Image.ANTIALIAS)
        img_data = np.asarray(img_resized)
        if is_grey:
            img_data = np.expand_dims(img_data, axis=2)
        images_np.append(img_data)

    images_np = np.asarray(images_np)
    images_np = (images_np - np.min(images_np)) / np.ptp(images_np)  # Normalization [0,1]
    return images_np

def save_images(path, all_images, prefix, labels, max_number_images=None, resize_dimensions=None, invert=False):

    if not os.path.exists(path):
        os.makedirs(path)

    img_idx = 0

    for img_np in all_images:
        img_np = np.squeeze(img_np)
        img = Image.fromarray((img_np * 255).astype('uint8'))
        if invert:
            img = ImageOps.invert(img)
        if resize_dimensions is not None:
            img = img.resize(resize_dimensions, Image.ANTIALIAS)
        img.save(path + '/' + prefix  + '_' + str(img_idx)+ '_' + labels[img_idx] + '.png')
        img_idx += 1

        if max_number_images is not None and img_idx == max_number_images:
            break

def load_dataset_patch(directory):
    images = []; labels = []
    count = 0

    for img_path in sorted(glob.glob(directory + "/*.png")):
        img = np.asarray(Image.open(img_path).convert('L'))
        images.append(img)
        if(count%100==0):
            print("Reading " + str(count) + " image")
        count = count + 1
        
        dir1 = img_path.split("/")
        dir2 = dir1[1]
        dir3 = dir2[:-4]
        labels.append(dir3)
        
    images_np=np.asarray(images).astype('float32')/255  # Normalization [0,1]
    images_final = np.expand_dims(images_np, axis=3)
    
    return images_final, labels

def load_dataset_full(directory):
    images = []; labels = []; mask = []
    count = 0

    for img_path in sorted(glob.glob(directory + "/*.png")):
        img = np.asarray(Image.open(img_path).convert('L'))
        if(count%100==0):
            print("Reading " + str(count) + " image")
        count = count + 1
        ret, thresh = cv2.threshold(img,20,255,cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
        num_labels, labels_im = cv2.connectedComponents(opening)
        values = []
        for value in range(1,num_labels):
            values.append(len(np.where(labels_im==value)[0]))
        
        final_mask = labels_im == np.where(values == np.amax(values))[0][0] +1
        
        images.append(img)
        mask.append(final_mask)

        dir1 = img_path.split("/")
        dir2 = dir1[1]
        dir3 = dir2[:-4]
        labels.append(dir3)
        
    images_np=np.asarray(images).astype('float32')/255  # Normalization [0,1]
    images_final = np.expand_dims(images_np, axis=3)
    mask = np.asarray(mask)
    
    return images_final, labels, mask
