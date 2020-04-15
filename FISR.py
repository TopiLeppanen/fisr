# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:40:48 2020

@author: Topi Leppänen

This file has the main superresolution function and an example of how it can
be used to upscale the image.
"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sklearn.feature_extraction.image as ski

import train

SIGMA_BLUR = 0.55
PATCH_SIZE = 5
PATCH_IDX = PATCH_SIZE//2
# Scaling factor
s = 1.5

#Tuning parameter to estimate the image noise variance. Kinda difficult to
#tune, this was found by just trying out a few different values and seeing when
#the weights distributed somewhat sensibly.
VARIANCE = 0.05

#Aggregate the 3x3 neighborhood around the lowresolution coordinate with weights
AGGREGATE = True
#Plot the y0 indexes as a image
PLOT_Y0_MAPPING = True

# Includes the transformer that does the mapping for y0 patch to a scalar 0-127
TRANSFORMER_FILE = "pca.sav"
# 1x128 vector of nabla f values. If only this filename is changed, the
# same transformer can be used without retraining.
C_FILE = "pca.npy"


# This function can implement any of the below superresolution algorithms.
# 'fisr' is the Fast Image Super Resolution, which is the most important and
# complex algorithm here.
def superresolution(im, method = 'fisr'):
    produceALL = method == 'all'
    produceFISR = method == 'fisr' or produceALL
    produce0ORDER = method == '0order' or produceALL
    produceBICUBIC = method == 'bicubic' or produceALL
    produceFREQ = method == 'freq' or produceALL
    
    # Produce the blurred version of lowresolution input and the interpolated
    # 'stretched' version of the input image.
    im_blur = cv2.GaussianBlur(im, (0,0), SIGMA_BLUR)
    im_interp = cv2.resize(im,(int(s*im.shape[1]), int(s*im.shape[0])), interpolation=cv2.INTER_CUBIC)
    
    size_lowres = im.shape
    size_highres = im_interp.shape
    
    #Extract full patches from highresolution images
    hr_coordx = np.array(range(PATCH_IDX, size_highres[1]-PATCH_IDX))
    hr_coordy = np.array(range(PATCH_IDX, size_highres[0]-PATCH_IDX))
    y_patch = np.array([im_interp[a-PATCH_IDX:a+PATCH_IDX+1,b-PATCH_IDX:b+PATCH_IDX+1]
                for a in hr_coordy
                for b in hr_coordx]) 
    
    #Extract full patches from lowresolution images
    lr_coordx = np.array(range(PATCH_IDX, size_lowres[1]-PATCH_IDX))
    lr_coordy = np.array(range(PATCH_IDX, size_lowres[0]-PATCH_IDX))
    y0_patch = np.array([im_blur[a-PATCH_IDX:a+PATCH_IDX+1,b-PATCH_IDX:b+PATCH_IDX+1]
                for a in lr_coordy
                for b in lr_coordx]) 
    x0_patch = np.array([im[a-PATCH_IDX:a+PATCH_IDX+1,b-PATCH_IDX:b+PATCH_IDX+1]
                for a in lr_coordy
                for b in lr_coordx]) 
    

    
    if produceFREQ or produce0ORDER:
        #Perform the frequency transfer here
        patch_idxs_lowres = train.get_lowres_indexes(y_patch.shape[0], size_highres, size_lowres) 
        x0_patch2 = x0_patch[patch_idxs_lowres]
        y0_patch2 = y0_patch[patch_idxs_lowres]
        x_patch_freq = y_patch + x0_patch2 - y0_patch2
    if produceFISR:
        if AGGREGATE:
            # Collect the 3x3 neighborhood to this large patch array
            x_patch_arr = np.zeros((9, y_patch.shape[0], y_patch.shape[1], y_patch.shape[2], y_patch.shape[3]), dtype=np.float32)
            weight_arr = np.zeros((9, y_patch.shape[0]), dtype=np.float64)
            for i in range(9):
                
                # This function finds the lowresolution patch indexes based
                # on image sizes and i, which is the offset for the
                # lowresolution coordinates
                patch_idxs_lowres = train.get_lowres_indexes(y_patch.shape[0], size_highres, size_lowres,i)
                
                # Find the lowresolution patches by indexing the previously
                # extracted full patch arrays
                tmp_y0 = y0_patch[patch_idxs_lowres]
                tmp_x0 = x0_patch[patch_idxs_lowres]
                
                # Find the nabla f values corresponding to y0 patches
                cvalues = train.get_c_values(tmp_y0, transformer, c)
                
                # Main first order regression function is evaluated
                x_patch_arr[i] = tmp_x0 + np.multiply(
                    cvalues[:,np.newaxis,np.newaxis,np.newaxis],y_patch-tmp_y0)
                
                
                if i == 4 and PLOT_Y0_MAPPING:
                    #Plot the y0 mapping image
                    cindexes = train.get_c_indexes(tmp_y0, transformer)
                    plt.figure()
                    plt.imshow(cindexes.reshape((size_highres[0]-4, size_highres[1]-4)).astype(np.uint8))
                    
                # Compute the weights for the patches with this offset
                tmp = np.power(y_patch-tmp_y0, 2)
                weight_arr[i] = np.sum(tmp, axis=(1,2,3))
                weight_arr[i] = np.exp(-weight_arr[i]/(2*VARIANCE))
                
            # Make weights sum up to one for each patch separately
            weight_norms = np.sum(weight_arr, axis=0)
            weight_arr = weight_arr / weight_norms
            
            # Aggregate the final patches with weighted sum
            x_patch_arr = weight_arr[:,:,np.newaxis,np.newaxis,np.newaxis] * x_patch_arr
            x_patch_sr = np.sum(x_patch_arr, axis=0)
            
        else:
            # No aggregation.
            # This is equivalent to above with the offset value of 4 (no shift)
            # and no weights
            patch_idxs_lowres = train.get_lowres_indexes(y_patch.shape[0], size_highres, size_lowres)
            y0_patch = ski.extract_patches_2d(im_blur, (PATCH_SIZE, PATCH_SIZE))          
            x0_patch= ski.extract_patches_2d(im, (PATCH_SIZE, PATCH_SIZE))   
            x0_patch = x0_patch[patch_idxs_lowres]
            y0_patch = y0_patch[patch_idxs_lowres]
            cvalues = train.get_c_values(y0_patch, transformer, c)
            x_patch_sr = x0_patch + np.multiply(cvalues[:,np.newaxis,np.newaxis,np.newaxis],y_patch-y0_patch)
           

    # Reconstruct the image from the patches.
    if produce0ORDER:
        im_zeroorder = ski.reconstruct_from_patches_2d(x0_patch2, size_highres)
    if produceFREQ:
        im_freq = ski.reconstruct_from_patches_2d(x_patch_freq, size_highres)
        im_freq = np.clip(im_freq,0,1)
    if produceFISR:
        im_sr = ski.reconstruct_from_patches_2d(x_patch_sr, size_highres)
        im_sr = np.clip(im_sr,0,1)
    im_interp = np.clip(im_interp,0,1)
              
    if produceALL:
        return [im_sr, im_zeroorder, im_freq, im_interp]
    if produceFISR:
        return im_sr
    if produce0ORDER:
        return im_zeroorder
    if produceFREQ:
        return im_freq
    if produceBICUBIC:
        return im_interp

tic = time.time()
if os.path.exists(TRANSFORMER_FILE):
    transformer = pickle.load(open(TRANSFORMER_FILE,'rb'))
    if os.path.exists(C_FILE):
        c = np.load(C_FILE)
    else:
        print("Mapping function not found! Recomputing...")
        [transformer, c] = train.train(TRANSFORMER_FILE, C_FILE)
else:
    print("transformer not found! Retraining...")
    [transformer, c] = train.train(TRANSFORMER_FILE, C_FILE)
    
toc = time.time()
print("Training elapsed: {}".format(toc-tic))

imgname = "INSERT_EXAMPLE_IMAGE_TO_UPSCALE_HERE"
im = Image.open(imgname)
im = np.array(im, dtype=np.float32)
im /= 255

tic = time.time()

im_sr = superresolution(im, method = 'fisr')
im_sr2 = superresolution(im_sr, method = 'fisr')
im_sr3 = superresolution(im_sr2, method = 'fisr')


toc = time.time()
print("Superresolution elapsed: {}".format(toc-tic))

plt.figure()
plt.imshow(im_sr3)