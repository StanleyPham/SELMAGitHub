# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:53:44 2020

@author: user
"""
#%%

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom

#%%

path = r'C:\Pulsate\Data\Jari\Essen_09\SELMA\ESSEN_SUB9_MALE_65Y___1.MR.0014.0001.2018.06.22.18.00.19.593750.32599405.IMA'
im = pydicom.dcmread(path).pixel_array

#%%

im2     = np.asarray(im > np.percentile(im, 40), dtype='uint8')
kernel  = np.ones((15,15), dtype='uint8')
im3     = cv2.erode(im2, kernel, 10)
# plt.imshow(im3)

kernel2  = np.ones((50,50), dtype='uint8')
im4     = cv2.dilate(im3, kernel2, 1)
# plt.imshow(im4)


#fill hole

im5     = np.copy(im4)
h,w     = im5.shape
mask    = np.zeros((h+2, w+2), np.uint8)

cv2.floodFill(im5, mask, (0,0), 1)
cv2.floodFill(im5, mask, (h-1,w-1), 1)
# plt.imshow(im5)

im6     = cv2.bitwise_not(im5)
im6     = im6 == np.max(im6)
# plt.imshow(im6)

im7     = im4 | im6
# plt.imshow(im7)

kernel3     = np.ones((80,80), dtype='uint8')
im8     = cv2.erode(im7, kernel3, 1)
plt.imshow(im8)

# plt.imshow(im*im8)
