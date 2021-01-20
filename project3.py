# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:28:19 2020

@author: Ryan
"""
import train, proj3
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from    skimage.measure    import    label,    regionprops,    moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from scipy import ndimage
from PIL import Image
from scipy import signal

from scipy import misc


i = 0
pkl_file = open("test_gt_py3.pkl", "rb")
mydict = pickle.load(pkl_file)
pkl_file.close()
classes = mydict[b'classes']
locations = mydict[b'locations']

proj3.project3("test.bmp", classes, locations)










