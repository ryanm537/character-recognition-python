# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:02:22 2020

@author: Ryan
"""
import train
import train
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


def project3(filename, classes, locations):
    Features = []
    CharList = []
    Locations = []
    
    train.train("a.bmp", Features, CharList);
    train.train("w.bmp", Features, CharList);
    train.train("o.bmp", Features, CharList);
    train.train("n.bmp", Features, CharList);
    train.train("d.bmp", Features, CharList);
    train.train("m.bmp", Features, CharList);
    train.train("p.bmp", Features, CharList);
    train.train("q.bmp", Features, CharList);
    train.train("r.bmp", Features, CharList);
    train.train("u.bmp", Features, CharList);
    
    #compute mean and standard deviation of features
    mean1 = 0
    mean2 = 0
    mean3 = 0
    mean4 = 0
    mean5 = 0
    mean6 = 0
    mean7 = 0
    for i in Features:
        mean1 += i[0]
        mean2 += i[1]
        mean3 += i[2]
        mean4 += i[3]
        mean5 += i[4]
        mean6 += i[5]
        mean7 += i[6]
    mean1 = mean1/len(Features)
    mean2 = mean2/len(Features)
    mean3 = mean3/len(Features)
    mean4 = mean4/len(Features)
    mean5 = mean5/len(Features)
    mean6 = mean6/len(Features)
    mean7 = mean7/len(Features)
    
    Features2 = np.array(Features).transpose()
    std1 = np.std(Features2[0])
    std2 = np.std(Features2[1])
    std3 = np.std(Features2[2])
    std4 = np.std(Features2[3])
    std5 = np.std(Features2[4])
    std6 = np.std(Features2[5])
    std7 = np.std(Features2[6])
    
    #normalization
    for i in Features:
        i[0] = (i[0] - mean1)/std1
        i[1] = (i[1] - mean2)/std2
        i[2] = (i[2] - mean3)/std3
        i[3] = (i[3] - mean4)/std4
        i[4] = (i[4] - mean5)/std5
        i[5] = (i[5] - mean6)/std6
        i[6] = (i[6] - mean7)/std7
    
    
    
    testImg = io.imread(filename);
    
    
    
    
    th = 200
    
    
    
    blurFilter = np.array([[1, 0, 0, 0, 0], 
                           [0, 1, 0, 0, 0], 
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])
    
    
    ndimage.convolve(testImg, blurFilter, testImg)
    #testing convolution
    
    
    img_binary = (testImg < th).astype(np.double)
    
    
    io.imshow(img_binary)
    plt.title('Binary Image')
    
    #dsiplaying component bounding boxes
    img_label = label(img_binary, background=0)
    
    
    
    
    regions = regionprops(img_label)
    ax = plt.gca()
    newFeatures = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if (maxc - minc > 15) & (maxr - minr > 15) & (maxc - minc < 65) & (maxr - minr < 65):
            ax.add_patch(Rectangle((minc-4,  minr-4),  maxc -minc+8,  maxr -minr+8, fill=False, edgecolor='red', linewidth=1))
            roi = img_binary[minr-4:maxr+4, minc-4:maxc+4]
            m = moments(roi)
            cc = m[0, 1] / m[0, 0]
            cr = m[1, 0] / m[0, 0]
            mu = moments_central(roi, center=(cr, cc))
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            newFeatures.append(hu)
            Locations.append([minr-4,maxr+4,minc-4,maxc+4])
        
        
    ax.set_title('Bounding Boxes')
    io.show()
    
    #normalizing newFeatures
    for i in newFeatures:
        i[0] = (i[0] - mean1)/std1
        i[1] = (i[1] - mean2)/std2
        i[2] = (i[2] - mean3)/std3
        i[3] = (i[3] - mean4)/std4
        i[4] = (i[4] - mean5)/std5
        i[5] = (i[5] - mean6)/std6
        i[6] = (i[6] - mean7)/std7
    
    #recognition on training data
    E = cdist(Features, newFeatures)
    io.imshow(E)
    plt.title('E Distance Matrix')
    io.show()
    E_index = np.argsort(E, axis=0)
    
    arrayForChars = []
    ii = 0
    seven = 0
    
    Candidates = []
    
    arrayForChars = []
    arrayForChars1 = []
    #print(CharList)
    #print(E_index)
    for i in E_index[0]:
        #print(i)
        arrayForChars1 = []
        arrayForChars1.append(CharList[E_index[0][ii]])
        arrayForChars1.append(CharList[E_index[1][ii]])
        arrayForChars1.append(CharList[E_index[2][ii]])
        arrayForChars1.append(CharList[E_index[3][ii]])
        arrayForChars1.append(CharList[E_index[4][ii]])
        arrayForChars1.append(CharList[E_index[5][ii]])
        arrayForChars1.append(CharList[E_index[6][ii]])
        #C = CharList[E[ii].tolist().index(i[0])]
        #C = CharList[i]
        #print(arrayForChars1)
        frequencies = [0, 0, 0, 0, 0, 0, 0]
        iii = 0
        for j in arrayForChars1:
            iv = 0
            for k in arrayForChars1:
                if arrayForChars1[iii] == arrayForChars1[iv]:
                    frequencies[iii] += 1;
                iv += 1
            iii += 1
        
        highestFrequency = 0;
        for j in frequencies:
            if highestFrequency < j:
                highestFrequency = j
        
        
        iii = 0
        for j in arrayForChars1:
            if frequencies[iii] == highestFrequency:
                arrayForChars.append(j)
                break
            iii += 1
        
        #print(i[0])]
        ii += 1
    #print(np.shape(E_index))
    io.imshow(img_binary)
    
    
    print(arrayForChars)
    
    i = 0
    hits = 0
    ax = plt.gca()
    for f in arrayForChars:
        j = 0
        for ff in locations:
            if (Locations[i][0] < locations[j][1])&(Locations[i][1] > locations[j][1]) & (Locations[i][2] < locations[j][0])&(Locations[i][3] > locations[j][0]):
                ax.text(Locations[i][2], Locations[i][0], arrayForChars[i],color = "white",fontSize = 20)
                ax.add_patch(Rectangle((Locations[i][2],  Locations[i][0]),  Locations[i][3] - Locations[i][2],  Locations[i][1] - Locations[i][0], fill = False, edgecolor='red', linewidth = 1))
                if classes[j] == arrayForChars[i]:
                    hits += 1
            j += 1
        i += 1
    # print(total)
    print("Recognition rate: " + str(hits / len(locations)))
    print("Components: " + str(len(Locations)))
    io.show()
