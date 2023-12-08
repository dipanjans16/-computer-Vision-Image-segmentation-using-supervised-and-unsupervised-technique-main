# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:59:21 2023

@author: Dipanjan
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
import scipy.misc
import scipy.ndimage
import skimage.filters
import sklearn.metrics


def Greyscale_converter(im_ADDRESS):
    # LOADING AND VISUALIZING DATA
    im_sto = imageio.imread(im_ADDRESS)
    return (255 - im_sto)

im_GSI=Greyscale_converter('cell_diagram.png')
binary_seperation=imageio.imread("predictions.png")


plt.imshow(im_GSI, cmap='gray')
plt.title("grayscale image")
plt.show()

plt.imshow(binary_seperation, cmap='gray')
plt.title("binary classification of image")
plt.show()



def visualize_thresholding(im_test):
    population_value,_ = np.histogram(im_test, bins=range(255))
    plt.plot(population_value)
    plt.title("pixel intensity plot")
    plt.xlabel("population")
    plt.ylabel("")
    plt.show()

visualize_thresholding(im_GSI)



def thresholding_using_Otsu(im_test):
    filter = scipy.ndimage.median_filter(im_test, size=10)
    " thresholding using Otsu "
    threshold = skimage.filters.threshold_otsu(filter)
    return np.uint8(filter> threshold) * 255

predicted_values=thresholding_using_Otsu(im_GSI)

binary_seperation = (binary_seperation).flatten().tolist()
predicted_values = (predicted_values).flatten().tolist()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(binary_seperation,predicted_values)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()



from sklearn.metrics import accuracy_score
print("accuracy of classification is", accuracy_score(binary_seperation,predicted_values))




"unsupervised"

import numpy as np
import matplotlib.pyplot as plt

import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
# import the image
from skimage import io


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

image = io.imread('Frame_1.png') 
plt.imshow(image)
image_slic = seg.slic(image,n_segments=200,compactness=5)
# label2rgb replaces each discrete label with the average interior color
image_show(color.label2rgb(image_slic, image, kind='avg'));


image_felzenszwalb = seg.felzenszwalb(image) 
image_show(image_felzenszwalb)
image_felzenszwalb_colored = color.label2rgb(image_felzenszwalb, image, kind='avg')
image_show(image_felzenszwalb_colored);
