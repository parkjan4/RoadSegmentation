# some functions are from given "mask_to_submission.py" and "segment_aerial_images.ipynb"
# some functions are also our own functions

import os
import numpy as np
import matplotlib.image as mpimg
import re
import sys
from PIL import Image
import keras.backend as K


"""FUNCTION FOR CREATING SUBMISSION"""
def create_submission(submission_filename, pred_array):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        counter = 0
        for img_index in range(1, 51):
            for j in range(38):
                for i in range(38):
                    f.write("{:03d}_{}_{},{}\n".format(img_index, 16*j, 16*i, pred_array[counter]))
                    counter += 1  

"""FUNCTION USED IN RUN.PY"""
def ensemble_model(predictions):
    """Takes predictions from several models 
    and outputs ensemble predictions based on majority voting rule"""
    
    # number of models in the ensemble model
    num_models = predictions.shape[1]
    
    sum_labels = np.zeros(predictions.shape[0])
    
    # add labels
    for c in range(num_models):
        sum_labels += predictions[:,c]
    
    # majority voting rule
    ensemble = (sum_labels >= (num_models/2)) * 1
    return ensemble


"""FUNCTIONS USED IN OUR CNN_MODEL.PY: SOME CUSTOM METRICS TO USE IN TRAINING PROCESS"""
def precision(y_true, y_pred):
    """Precision metric"""
    return K.sum(y_true * y_pred ) / K.sum(y_pred)

def recall(y_true, y_pred):
    """Recall metric. Recall is the same as true positive rate."""
    return K.sum(y_true * y_pred) / K.sum(y_true)

def fmeasure(y_true, y_pred):
    """Computes the f-measure"""
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    return 2*precision_*recall_ / (precision_ + recall_)
                    

"""GIVEN HELPER FUNCTIONS"""
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
    
# Compute features for each image patch
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

    
"""FUNCTIONS FOR DATA PRE-PROCESSING"""
def image_padding(X, padding):
    """Add padding to 16x16 patch images
    Input: 
        X: either train or test images of dimensions (?, 16, 16, 3) if RGB, otherwise (?, 16, 16)
    Output:
        X_pad: images of dimensions (?, 16+2*padding, 16+2*padding, 3) if RGB, otherwise (?, 16+2*padding, 16+2*padding)
    """
    
    # groundtruth images
    if len(X.shape) < 4:
        X_pad = np.empty((X.shape[0], X.shape[1]+2*padding, X.shape[2]+2*padding))
        for n in range(X.shape[0]):
            # Create mirror images of the original 16x16 patch image
            X_pad[n] = np.pad(X[n], ((padding, padding), (padding, padding)), 'reflect')
            
    # RGB images
    else:
        X_pad = np.empty((X.shape[0], X.shape[1]+2*padding, X.shape[2]+2*padding, X.shape[3]))
        for n in range(X.shape[0]):
            # Create mirror images of the original 16x16 patch image
            X_pad[n] = np.pad(X[n], ((padding, padding), (padding, padding), (0,0)), 'reflect')

    return X_pad


def to_onehot(labels,nclasses):
    '''
    Convert labels to "one-hot" format.
    >>> a = [0,1,2,3]
    >>> to_onehot(a,5)
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.]])
    '''
    outlabels = np.zeros((len(labels),nclasses))
    for i,l in enumerate(labels):
        outlabels[i,l] = 1
    return outlabels


def class_balancing(X, onehot_lab, labels):
    """ Undersample overrepresented class (background) so that road to non-road patches are in 1:1 ratio
    Input:
        X: train or test image
        onehot_lab: onehot coded labels or each image in X
    Output:
        X: image data with balanced class ratio. Will have less data than original X
    """
    
    c0 = 0
    c1 = 0
    for i in range(len(onehot_lab)):
        if onehot_lab[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print ('----------Balancing training data...----------')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(onehot_lab) if j[0] == 1]
    idx1 = [i for i, j in enumerate(onehot_lab) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print('Original training data size: {s:}'.format(s=X.shape))
    X = X[new_indices,:,:,:]
    onehot_lab = onehot_lab[new_indices]
    labels = labels[new_indices]

    size = onehot_lab.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(onehot_lab)):
        if onehot_lab[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
    print('New training data size: {s:}'.format(s=X.shape))

    return X, onehot_lab, labels


"""CROSS VALIDATION"""
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(model, test, onehot_te, k_fold):
    indices = build_k_indices(test, k_fold, seed=1)
    CV_results = np.empty((k_fold, 3))
    for k in range(k_fold):
        CV_results[k,:] = model.evaluate(test[indices[k],:,:,:], onehot_te[indices[k],:])
    
    return CV_results