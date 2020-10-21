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
import SELMAInterpolate
#%%
im      = sitk.GetArrayFromImage(sitk.ReadImage(
r'C:\Pulsate\Data\Tests\C001\t1\StructuralNii\s_T1w_3D_TFE_pulsate_401.nii'))
seg     = sitk.GetArrayFromImage(sitk.ReadImage(
r'C:\Pulsate\Data\Tests\C001\t1\Segmentation\c2s_T1w_3D_TFE_pulsate_401.nii'))
#%%
im  = np.flip(im, 1)
seg = np.flip(seg, 1)

#%%

pcafn = r'C:\Pulsate\Data\Tests\C001\8122061_06_01_tWM 2DPCAtoneTFE2 SENSE1.5.dcm'
t1fn  = r'C:\Pulsate\Data\Tests\C001\t1\8122061_04_01_s T1w 3D TFE pulsate.dcm'

#%%    

pca = pydicom.dcmread(pcafn)
t1  = pydicom.dcmread(t1fn)
Mpca, Rpca  = SELMAInterpolate.getTransMatrix(pca)
Mt1, Rt1    = SELMAInterpolate.getTransMatrix(t1)
M           = np.dot(np.linalg.inv(Mt1), Mpca)
pcaShape    = pca.pixel_array.shape

#%%

t1Slice   = SELMAInterpolate.doInterpolation(M, im, pcaShape)
plt.imshow(t1Slice)

#%%
maskSlice   = SELMAInterpolate.doInterpolation(M, seg, pcaShape)
plt.imshow(maskSlice)
