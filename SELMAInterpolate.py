#!/usr/bin/env python

"""
This static module contains the following functions:
    
+ :function:`usableImagePosition`
+ :function:`usableImageOrientation`
+ :function:`rotXAxis`
+ :function:`rotYAxis`
+ :function:`rotZAxis`
+ :function:`findAngleDiff`
+ :function:`getInterpolationVariables`

"""

import numpy as np
import math


def usableImagePosition(dcm):
    pos     = dcm[0x5200, 0x9230][0][0x0020, 0x9113][0].ImagePositionPatient
    return np.array([float(pos[0]), float(pos[1]), float(pos[2])])

def usableImageOrientation(orr):
    xx, xy, xz, yx, yy, yz = orr
    row   = np.asarray([float(xx), float(xy), float(xz)])
    col   = np.asarray([float(yx), float(yy), float(yz)])
    
    return row, col

def rotXAxis(r, theta = math.pi/2 ):
    '''
    Rotates a 3x1 vector around the x-axis for 90 degrees
    '''
    
    rotMat  = np.asarray([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]])
    
    return  np.dot(rotMat, r)


def rotYAxis(r):
    '''
    Rotates a 3x1 vector around the y-axis for 90 degrees
    '''
    
    rotMat  = np.asarray([[0, 0, 1],
                          [0, 1, 0],
                          [-1,0, 0]])
    
    return  np.dot(rotMat, r)

def rotZAxis(r):
    '''
    Rotates a 3x1 vector around the z-axis for 90 degrees
    '''
    
    rotMat  = np.asarray([[0, -1,0],
                          [1, 0, 0],
                          [0, 0, 1]])
    
    return  np.dot(rotMat, r)


def findAngleDiff(orr1, orr2):
    '''
    Finds the difference in radians between two directional cosine vectors 
    along each axis.
    Subtracts pi/2 to get the angle from zero.
    '''
    
    dTheta  = math.acos(orr1[0] - orr2[0]) - math.acos(0)
    dPhi    = math.acos(orr1[1] - orr2[1]) - math.acos(0)
    dKsi    = math.acos(orr1[2] - orr2[2]) - math.acos(0)
    
    return np.array([dTheta, dPhi, dKsi])



def getInterpolationVariables(dcm, rescaledImage = None):
    """
    Constructs a list of the real-world-coordinates of dicom file, as well
    as a list of the values. 
    The function is invariant to the orientation of the image.
    
    Input:
        dcm             -   dicom header; the output from pydicom.dcmread
        rescaledImage (optional) - the rescaled Dicom image array. 
        
    Returns:
        x   - list of all the real-world x-coordinates
        y   - list of all the real-world y-coordinates
        z   - list of all the real-world z-coordinates
        vals- list containing the rescaled values at each rwc. (only if 
        rescaledImage is provided).
    """
    
    
    #First get the necessary variables
    pixelSpacing = dcm[0x5200, 0x9230][0][0x0028, 0x9110][0][0x0028, 0x0030].value
    if rescaledImage is not None:
        shape   = rescaledImage.shape
    else:
        shape        = dcm.pixel_array.shape
    row_orr, col_orr    = usableImageOrientation(
            dcm[0x5200, 0x9230][0][0x0020, 0x9116][0].ImageOrientationPatient)
    
    # Preallocate Output variables
    x       = np.zeros(shape[0]*shape[1]*shape[2])
    y       = np.zeros(shape[0]*shape[1]*shape[2])
    z       = np.zeros(shape[0]*shape[1]*shape[2])
    vals    = np.zeros(shape[0]*shape[1]*shape[2])

    #Iterate over slices
    for i in range(shape[0]):
        #Find top left voxel position of that slice
        r0  = dcm[0x5200, 0x9230][i][0x0020, 0x9113][0].ImagePositionPatient
        
        #Find rwc for all three coordinate directions of every voxel in the 
        #slice. This is done by taking the position of the top left voxel of 
        #this slice (r0) and propagating in the row and column directions of
        #the slice. The directional cosines (row_orr, col_orr) are used to
        #determine how the rows and columns are oriented with respect to 
        #all three coordinate axes.
        
        #A 2d matrix is made for each coordinate axis showing the position
        #of each voxel in the current slice relative to the top left voxel.
        #The top left voxel position is then added to find the real world
        #coordinate of every voxel.
        
        #These coordinates are then added to the output lists for the x,y and z
        #axes.
        
        #first coordinate axis
        c01 = np.ones([shape[1], shape[2]]) * (np.arange(0,shape[1]) * 
                                                    pixelSpacing[0] * 
                                                    row_orr[0])
        c02 = (np.arange(0,shape[2]) * 
              pixelSpacing[1] * 
              col_orr[0])    * np.ones([shape[1], shape[2]])
        c02 = np.transpose(c02)
        
        c0  = r0[0] + c01 + c02
        
        #second coordinate
        c11 = np.ones([shape[1], shape[2]]) * (np.arange(0,shape[1]) * 
                                                    pixelSpacing[0] * 
                                                    row_orr[1])
        c12 = (np.arange(0,shape[2]) * 
              pixelSpacing[1] * 
              col_orr[1])    * np.ones([shape[1], shape[2]])
        c12 = np.transpose(c12)
        
        c1  = r0[1] + c11 + c12
        
        #third coordinate
        c21 = np.ones([shape[1], shape[2]]) * (np.arange(0,shape[1]) * 
                                                    pixelSpacing[0] * 
                                                    row_orr[2])
        c22 = (np.arange(0,shape[2]) * 
              pixelSpacing[1] * 
              col_orr[2])    * np.ones([shape[1], shape[2]])
        c22 = np.transpose(c22)
        
        c2  = r0[2] + c21 + c22
        
        #Create x,y,z lists for this slice
        points    = np.array([c0,c1,c2])
        #Reshape coordinates to list
        [xslice, yslice, zslice]  = np.reshape(points, (3, -1))
        
        #Create value list for this slice        
        if rescaledImage is not None:
            imSlice     = rescaledImage[i]
            valSlice    = np.reshape(imSlice, -1)
        else:
            valSlice    = []
        
        #Add to output
        x[i * shape[1]*shape[2] : (i+1)*shape[1]*shape[2]] = xslice
        y[i * shape[1]*shape[2] : (i+1)*shape[1]*shape[2]] = yslice
        z[i * shape[1]*shape[2] : (i+1)*shape[1]*shape[2]] = zslice
        vals[i * shape[1]*shape[2] : (i+1)*shape[1]*shape[2]] = valSlice
        
    return x, y, z, vals
    
    