# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:54:40 2020

@author: user
"""
import numpy as np
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

#%%
t1fn = r'C:\Pulsate\Data\Tests\C001\t1\8122061_04_01_s T1w 3D TFE pulsate.dcm';
pcafn = r'C:\Pulsate\Data\Tests\C001\8122061_06_01_tWM 2DPCAtoneTFE2 SENSE1.5.dcm';

t1info = pydicom.dcmread(t1fn);
pcainfo = pydicom.dcmread(pcafn);

#%% Get transform matrix
def getTransMatrix(info):
    ipp = info.PerFrameFunctionalGroupsSequence[0].\
                PlanePositionSequence[0].\
                ImagePositionPatient
    ipp = [float(ipp[0]), float(ipp[1]), float(ipp[2])]
    iop = info.PerFrameFunctionalGroupsSequence[0].\
                PlaneOrientationSequence[0].\
                ImageOrientationPatient
    ps  = info.PerFrameFunctionalGroupsSequence[0].\
        PixelMeasuresSequence[0].\
            PixelSpacing
    ps  = [float(ps[0]), float(ps[1])]
    st  = float(info.
              PerFrameFunctionalGroupsSequence[0]
              [0x2005,0x140f][0].SliceThickness)


    #%Translate to put top left pixel at ImagePositionPatient
    Tipp = [[1, 0, 0, ipp[0]],
            [0, 1, 0, ipp[1]],
            [0, 0, 1, ipp[2]],
            [0, 0, 0, 1] ]
    
    #Rotate into patient coordinate system using direction 
    #cosines from ImagOrientationPatient
    r   = [float(iop[0]), float(iop[1]), float(iop[2])]
    c   = [float(iop[3]), float(iop[4]), float(iop[5])]
    s   = np.cross(r,c)    
    R   = [[r[0], c[0], s[0], 0],
           [r[1], c[1], s[1], 0],
           [r[2], c[2], s[2], 0],
           [0,  0,  0,  1]]
    
    #Scale using PixelSpacing
    if info.MRAcquisitionType != '2D':
        st = float(info.SpacingBetweenSlices)
    S = [  [ps[1], 0, 0, 0],
           [0, ps[0], 0, 0],
           [0, 0, st, 0],
           [0, 0, 0, 1]]
           
    
    #Shift image to make top left voxel centre at (0,0,0)
    T0 = [  [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]
    
    #Construct transformation matrix
    M = np.dot(
            np.dot(
                np.dot(Tipp, R), S), T0)
    
    return M, R
    

    

#%%

Mpca, Rpca  = getTransMatrix(pcainfo)
Mt1, Rt1    = getTransMatrix(t1info)

M   = np.dot(np.linalg.inv(Mt1), Mpca)
Rot = np.dot(np.linalg.inv(Rt1), Rpca)




#%% swap axes for x and y 

t1toqfnew = np.zeros((4,4))
t1toqfnew[:,0] = M[:,1]
t1toqfnew[:,1] = M[:,0]
t1toqfnew[:,2] = M[:,2]
t1toqfnew[:,3] = M[:,3]
#%%

newt1niifilename = \
r'C:\Pulsate\Data\Tests\C001\t1\StructuralNii\s_T1w_3D_TFE_pulsate_401.nii'

t1nii = sitk.GetArrayFromImage(sitk.ReadImage(newt1niifilename))

#Switch axes to x,y,z so that it is the same as matlab
#There's probably a way to combine the next two steps into one, but this works
t1nii = np.swapaxes(t1nii, 0,2)

#Reorient the image such that it maches the LPH orientation of the PCA
t1_3D = np.flip(t1nii,1)
t1_3D = np.flip(t1_3D,2)
t1_3D = np.swapaxes(t1_3D, 1,0)

#%% test
plt.imshow(t1_3D[100,:,:])

#%% get coordinate ranges
corrcoorx = -0.5
corrcoory = -1.5
corrcoorz = -1.5
Zrange = 1

range_x = np.arange(1,pcainfo.pixel_array.shape[2] + 1) + corrcoorx
range_y = np.arange(1,pcainfo.pixel_array.shape[1] + 1) + corrcoory
range_z = Zrange + corrcoorz;


#%% copy of affine3Dlennart

M = t1toqfnew + [[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]]

dV =  [-0.5, -0.5, -0.5]

#x = 1:1280, y = 0:1279, z = 0

range_x = range_x - dV[0]
range_y = range_y - dV[1]
range_z = range_z - dV[2]

yg, xg, zg = np.meshgrid(range_y,range_x,range_z)

xyz = np.asarray([
       np.reshape(xg,-1),
       np.reshape(yg,-1),
       np.reshape(zg, -1),
       np.ones(len(xg)**2)])

#transform
uvw = np.dot(M, xyz)
uvw = np.transpose(uvw[:3,:])

# #sample
# xi = np.reshape(uvw[:,0], xg.shape[:2])
# yi = np.reshape(uvw[:,1], xg.shape[:2])
# zi = np.reshape(uvw[:,2], xg.shape[:2])

# xi -= 0.5
# yi -= 0.5
# zi -= 0.5

# interpolate


#%% interpolate
#Here we vary from lennart's method, as we have to use 
#scipy.RegularGridInterpolator

xi = uvw[:,0]
yi = uvw[:,1]
zi = uvw[:,2]

index = (xi > 0) * (xi < t1_3D.shape[0] - 1) * \
        (yi > 0) * (yi < t1_3D.shape[1] - 1) * \
        (zi > 0) * (zi < t1_3D.shape[2] - 1)

xi = xi[index]
yi = yi[index]
zi = zi[index]


# Why is this different?
pts = np.transpose((yi, xi, zi))   

x = np.arange(0,t1_3D.shape[0])
y = np.arange(0,t1_3D.shape[1])
z = np.arange(0,t1_3D.shape[2])

inter = RegularGridInterpolator((x,y,z), t1_3D)
inp     = inter(pts)

res = np.zeros(len(index))
res[index] = inp
res = np.reshape(res, xg.shape[:2])

plt.imshow(res)



