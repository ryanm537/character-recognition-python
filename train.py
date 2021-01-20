# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:28:19 2020

@author: Ryan
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from    skimage.measure    import    label,    regionprops,    moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from scipy import ndimage
from PIL import Image, ImageEnhance
import skimage
from skimage.viewer import ImageViewer
import sys

def train(filename, Features, CharList):

#reading an image file
    img = io.imread(filename);
    
	#visualizing an image/matrix
    
    #testing convolution
  
    
    blurFilter = np.array([[1, 0, 0, 0, 0], 
                           [0, 1, 0, 0, 0], 
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])


    ndimage.convolve(img, blurFilter, img)
        
    io.imshow(img)
    plt.title('Original Image')
    io.show()
    
	#image histogram
    hist = exposure.histogram(img)
    plt.bar(hist[1], hist[0])
    plt.title('Histogram')
    plt.show()
    
    
	#Binarization by Thresholding
    th = 200
    img_binary = (img < th).astype(np.double)
    
    
    
    
	#displaying binary image
    io.imshow(img_binary)
    plt.title('Binary Image')
    io.show()

	#extracting characters and their features
    img_label = label(img_binary, background = 0)
    io.imshow(img_label)
    plt.title('Labeled Image')
    io.show()
    
    
    
	#storing features
    #Features=[]
    #CharList = []
    
	#displaying component bounding boxes
    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if (maxc - minc > 15) & (maxr - minr > 15) & (maxc - minc < 65) & (maxr - minr < 65):
            ax.add_patch(Rectangle((minc-4, minr-4), maxc - minc+8, maxr - minr+8, fill = False, edgecolor = 'red', linewidth = 1))
       		#computing hu moments and removing small components
            roi = img_binary[minr-4:maxr+4, minc-4:maxc+4]
            m =  moments(roi)
            cc = m[0,1] / m[0,0]
            cr = m[1, 0] / m[0, 0]
            mu = moments_central(roi, center=(cr, cc))
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            Features.append(hu)
            CharList.append(filename[0])
            ax.set_title('Bounding Boxes')
    io.show()
    
    
#train(train)






