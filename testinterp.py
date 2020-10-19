# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:00:50 2020

@author: user
"""


#%% test shit
import numpy as np
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
#%%
pca     = np.load(r'C:\Pulsate\Data\Tests\C001\t1\testPCAgrid.npy')
im      = sitk.GetArrayFromImage(sitk.ReadImage(
r'C:\Pulsate\Data\Tests\C001\t1\StructuralNii\s_T1w_3D_TFE_pulsate_401.nii'))
seg     = sitk.GetArrayFromImage(sitk.ReadImage(
r'C:\Pulsate\Data\Tests\C001\t1\Segmentation\c2s_T1w_3D_TFE_pulsate_401.nii'))

#%%


plt.imshow(im[90])
plt.imshow(seg[90])

#%%

im2 = np.zeros(im.shape)
im2 += im

#%%

for i in range(len(pca)):
    r = pca[i]
    x = int(r[2])
    y = int(r[1])
    z = int(r[0])
    im2[z,y,x] = 1000
    
#%%

plt.imshow(im2[90])

#%%

x = np.arange(0,320)
y = np.arange(0,320)
z = np.arange(0,190)

xi = []
yi = []
zi = []

for i in range(320):
    for j in range(320):
        zi.append(190/320 *j * 0.99)
        yi.append(318 - i/2)
        xi.append(i*0.99)


#%%
pts = np.transpose((zi, yi, xi))        

#%%

inter = RegularGridInterpolator((z,y,x), im)
inp     = inter(pts)
res     = np.reshape(inp, (320,320))
plt.imshow(res)

#%%

x = []
y = []
z = []

for i in range(0,len(pca),1000):
    r = pca[i]
    x.append(int(r[2]))
    y.append(int(r[1]))
    z.append(int(r[0]))
#%%    

