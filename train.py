# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:06:41 2020

@author: Topi Leppänen
This implements the functions to train the FISR.
Also some other utility functions.
This requires that the training data is found from the TRAINING_DATA_PATH
Berkeley image segmentation database was used in testing, but it's not provided
here.
"""
import os, time
import pickle
import numpy as np
import cv2
from PIL import Image
import sklearn.feature_extraction.image as ski
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# !!! Training data directory !!!
TRAINING_DATA_DIR = "../BSR/BSDS500/data/images"

# These must be the same as in the FISR.py.
SIGMA_BLUR = 0.55
PATCH_SIZE = 5
PATCH_IDX = PATCH_SIZE//2
s = 1.5

# This is the amount of random patches extracted per image when estimating y0
# mapping function
TRAINING_PATCH_COUNT = 1000

# This is used when extracting patches for nabla f function estimation.
# Value of 1 is equivalent to extracting all the overlapping patches from each
# image.
TRAINING_PATCH_STRIDE = 30

# Number of clusters in y0. Also the number of pieces in the piece-wise function
# nabla f.
C_COUNT = 128


# Returns the nabla f values corresponding to the patches
def get_c_values(patches, transformer, c):    
    predictions = get_c_indexes(patches,transformer)    
    return c[predictions]

# Returns the c index values corresponding to the patches
def get_c_indexes(patches, transformer):
          
    # Flatten the patches into a vector
    patches = patches.reshape(patches.shape[0],-1)
        
    # Two part transform, first PCA transform and then KMeans prediction.
    patches = transformer[0].transform(patches)
    predictions = transformer[1].predict(patches)
    
    return predictions

# Generates the lowresolution coordinates corresponding to the range
# of highresolution patch indexes from 0 to patch_count.
def highres_idxs_to_low_coords(patch_count, size_highres, size_lowres):
    patch_idxs = np.arange(patch_count)
    
    # Convert the patch indexes to high resolution coordinates
    ycoordy = (patch_idxs // (size_highres[1]-2*PATCH_IDX)) + PATCH_IDX
    ycoordx = (patch_idxs % (size_highres[1]-2*PATCH_IDX)) + PATCH_IDX
    
    # Convert the high resolution coordinates to lowresolution coordinates
    ycoordy = np.floor(ycoordy / s + 0.5)
    ycoordx = np.floor(ycoordx / s + 0.5)
    
    return [ycoordy.astype(int), ycoordx.astype(int)]
    
# Generates the lowresolution patch indexes corresponding to the range of
# highresolution patch indexes from 0 to patch_count. Uses the above function
# to get the centered lowresolution coordinates.
# The direcion argument can be used to implement the lowresolution coordinate
# offset of +-1 to any direction:
#  0  1  2
#  3  4  5
#  6  7  8
    
def get_lowres_indexes(patch_count, size_highres, size_lowres, direction=4):
    
    [ycoordy, ycoordx] = highres_idxs_to_low_coords(patch_count, size_highres, size_lowres)
 
    #these map direction value to correct coordinate offset values
    offset_v = direction // 3 - 1
    offset_h = direction % 3 - 1
    
    # Add the offset and clamp the coordinates to PATCH IDX away from image borders
    ycoordy = np.clip(ycoordy + offset_v, PATCH_IDX, size_lowres[0]-1-PATCH_IDX)
    ycoordx = np.clip(ycoordx + offset_h, PATCH_IDX, size_lowres[1]-1-PATCH_IDX)
    
    #Convert from coordinates to patch index space
    patch_idxs_lowres = (size_lowres[1]-2*PATCH_IDX)*(ycoordy-PATCH_IDX) + ycoordx - PATCH_IDX
    
    # Also clamp the patch indices to valid values
    patch_idxs_lowres = np.clip(patch_idxs_lowres, 0, (size_lowres[0]-2*PATCH_IDX)*(size_lowres[1]-2*PATCH_IDX)-1)
    return patch_idxs_lowres.astype(int)

# This function aggregates the aTa and bTa into an array As
# Is used to learn the nabla f values.
def get_ab(imgname,transformer, As):
    gt = Image.open(imgname)

    # ground truth high resolution image       
    gt = np.array(gt, dtype=np.float32)
    gt /= 255
    
    im = cv2.resize(gt,(int(np.floor(gt.shape[1]/s)), int(np.floor(gt.shape[0]/s))), interpolation=cv2.INTER_AREA)   
    im_blur = cv2.GaussianBlur(im, (0,0), SIGMA_BLUR)   
    im_interp = cv2.resize(im,(gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        
    size_lowres = im.shape
    size_highres = im_interp.shape
    
    # Extract patches in a regular grid with the wanted STRIDE. Avoids the edge
    # areas to prevent the clamping of coordinates in the training phase.    
    hr_coordx = np.array(range(PATCH_SIZE, size_highres[1]-PATCH_SIZE, TRAINING_PATCH_STRIDE))
    hr_coordy = np.array(range(PATCH_SIZE, size_highres[0]-PATCH_SIZE, TRAINING_PATCH_STRIDE))
        
    y_patch = np.array([im_interp[a-PATCH_IDX:a+PATCH_IDX+1,b-PATCH_IDX:b+PATCH_IDX+1]
                for a in hr_coordy
                for b in hr_coordx])
    x_patch = np.array([gt[a-PATCH_IDX:a+PATCH_IDX+1,b-PATCH_IDX:b+PATCH_IDX+1]
            for a in hr_coordy
            for b in hr_coordx])
    
    # Convert highresolution coordinates to lowresolution and extract the
    # corresponding patches.
    lr_coordx = np.floor(hr_coordx / s + 0.5).astype(int)
    lr_coordy = np.floor(hr_coordy / s + 0.5).astype(int)
    lr_coordx = np.clip(lr_coordx, PATCH_IDX, size_lowres[1]-1-PATCH_IDX)
    lr_coordy = np.clip(lr_coordy, PATCH_IDX, size_lowres[0]-1-PATCH_IDX)  
    
    y0_patch = np.array([im_blur[a-PATCH_IDX:a+PATCH_IDX+1,b-PATCH_IDX:b+PATCH_IDX+1]
                         for a in lr_coordy
                         for b in lr_coordx])
    
    x0_patch = np.array([im[a-PATCH_IDX:a+PATCH_IDX+1,b-PATCH_IDX:b+PATCH_IDX+1]
                         for a in lr_coordy
                         for b in lr_coordx])
           
    # Find out the y0 mapping of the training data
    c_idxs = get_c_indexes(y0_patch, transformer)
          
    
    tmp_a = y_patch - y0_patch
    tmp_b = x_patch - x0_patch
                 
    # Split the aTa and bTa values based on the c index value to the correct
    # buckets
    for idx in range(0,y_patch.shape[0]):
        As[c_idxs[idx],0] +=  np.dot(tmp_b[idx].ravel() , tmp_a[idx].ravel())
        As[c_idxs[idx],1] +=  np.dot(tmp_a[idx].ravel() , tmp_a[idx].ravel())
        
    return As
  
# Collects the y0 patches to be used for learning of y0 mapping function.
def get_y0(imgname):
    gt = Image.open(imgname)
    im = gt.resize((int(np.floor(gt.size[0]/s)), int(np.floor(gt.size[1]/s))))   
    im = np.array(im, dtype=np.float32)
    im /= 255
    
    im_blur = cv2.GaussianBlur(np.array(im), (0,0), SIGMA_BLUR).astype(np.float32)
    y0_patch = ski.extract_patches_2d(im_blur, (PATCH_SIZE, PATCH_SIZE), max_patches=TRAINING_PATCH_COUNT)
        
    return y0_patch.reshape(y0_patch.shape[0], -1)

# Main training function which can be used to learn both the y0 mapping "transformer"
# and the nabla f values, array "c".
# The y0 mapping is learned by collecting all the values in a huge array.
# This is to make the implementation simpler. To improve, this should be done
# incrementally to save memory.
def train(transformer_file, c_file, scoped = False):
    data_dir = TRAINING_DATA_DIR
    
    # Check if the transformer file with given name already exists.
    if not os.path.exists(transformer_file):
        tic = time.time()
        count=0
        # Goes through all the jpg images found in the given directory.
        for root, dirs, files in os.walk(data_dir):
                for name in files:
                    if name.endswith(".jpg"):
                        count +=1
                        tmp = get_y0(os.path.join(root,name))                      
                        if count == 1:
                            y0_patches = tmp
                        else:
                            y0_patches = np.append(y0_patches, tmp, axis=0)
                       
                      
        toc = time.time()
        print("Fetching y0: {} s".format(toc-tic))
        tic = time.time()                 
                            
        # Transformer is collected into this np.array
        # There's 2 parts to it, the PCA and KMeans
        transformer = []
        
        # Fit the PCA
        transformer.append(PCA(n_components=8))
        tmp = transformer[0].fit_transform(y0_patches)
                
        toc = time.time()
        print("Fitting PCA: {} s".format(toc-tic))
        tic = time.time()
        
        # Fit the KMeans
        transformer.append(KMeans(n_clusters=C_COUNT))
        transformer[1].fit(tmp)
        
        # Save the transformer combination
        pickle.dump(transformer, open(transformer_file, 'wb'))
        
        toc = time.time()
        print("Fitting KMeans: {} s".format(toc-tic))
    else:
        # IF the transformer file is already found, just use it.
        transformer = pickle.load(open(transformer_file,'rb'))
    
    tic = time.time()
    
    # aTa and bTa sums are aggregated in this array
    As = np.zeros((C_COUNT,2))
    for root, dirs, files in os.walk(data_dir):
            for name in files:
                if name.endswith(".jpg"):
                    As = get_ab(os.path.join(root,name), transformer, As)                       
    
    toc = time.time()
    
    c =np.zeros((C_COUNT), dtype=np.float32)
    # Finally divide the sums to get the LS solution for each c index.
    for k in range(C_COUNT):
        c[k] = As[k,0] / As[k,1]
        
    np.save(c_file, c)
    
    print("Learning the f gradient: {} s".format(toc-tic))
    
    return [transformer, c]
    

