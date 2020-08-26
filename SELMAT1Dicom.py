#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`SELMASegmentation`

"""


# ====================================================================

import SELMADicom
import SELMAInterpolate

import os
import pydicom
import SimpleITK as sitk
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matlab.engine


class SELMAT1Dicom(SELMADicom.SELMADicom):
    """
    This class deals with t1 segmentation
    """
    
    def __init__(self, dcmFilename, pcaDcm):
        """
        Load & rescale the T1 & interpolate correct slice
        """
        self._dcmFilename   = dcmFilename
        self._dcm           = pydicom.dcmread(self._dcmFilename)
        self._pcaDcm        = pcaDcm
        
        #############################################################
        #Declare some variables for use in interpolating / segmenting
                
        #T1 properties
        self._manufacturer  = None
        self._frames        = None
        self._numFrames     = None
        self._t1Slice       = None
        self._magFrameIndex = None
        
        #interpolating properties 
        self._x             = None
        self._y             = None
        self._z             = None
        self._xi            = None
        self._yi            = None
        self._zi            = None
        self._idx           = None
        self._pcaGrid       = None
        
        # Segmentation properties
        self._maskSlice     = None
        self._segmentation  = None
        
        #############################################################
        
        #Get only the magnitude frames from the T1
        self.findManufacturer()
        self.findMagnitudeFrames()
        
        #Interpolate the t1 slice
        self.findT1Grid()
        self.findInterpolatingGrid()
        self.interpolateT1Slice()
        
        
        #TODO: only use the right frames
        
    '''Public'''
    def getSegmentationMask(self):
        """
        Walks through the various functions for constructing a T1 segmentation.
        
        -Find libraries (SPM & dcm2nii)
        -Launch matlab engine
        -Run matlab script to convert to .nii & segment
        -Remove unnecessary files
        -Interpolate segmentation slice from brainmask           
            
        Returns:
            self._segmentation; the interpolated slice of the brainmask
        """
        libraries   =   SELMAInterpolate.getLibraries()
        spm         =   libraries[0]
        dcm2nii     =   libraries[1]
        
        eng         = matlab.engine.start_matlab()
        out         =   eng.spmSegment(self._dcmFilename, 
                                       spm, 
                                       dcm2nii)
        
        #Remove all unnecessary files that were generated
        for file in out['remove']:
            os.remove(file)
        os.remove(out['gm'])
        
        #Load the WM segmentation
        im  = sitk.ReadImage(out['wm'])
        im  = sitk.GetArrayFromImage(im)
        print(im.shape, np.unique(im))
        self._segmentation     = im
        
        #Create interpolated slice
        self.interpolateMaskSlice()
        
        return self._maskSlice

        
        
    
    def getFrames(self):
        """
            Override of SELMADicom getFrames function. Only returns the 
            interpolated pca slice. 
        """
        
        return self._t1Slice
    
    
    
    '''Private'''
    
    def findManufacturer(self):
        """
            Finds the manufacturer in the dicom tags
        """
        self._manufacturer  = self._dcm[0x8, 0x70].value
    
    def findMagnitudeFrames(self):
        """
            Finds the magnitude frames in the T1 dicom and stores the indices.        
        """
        
        self._magFrameIndex     = []
        
        if self._manufacturer   == "Philips Medical Systems": 
            for i in range(len(self._dcm.pixel_array)):
                dcmFrameAddress             = 0x5200, 0x9230
                dcmPrivateCreatorAddress    = 0x2005, 0x140f
                dcmImageTypeAddress         = 0x0008, 0x0008
                
                frameType   = self._dcm[dcmFrameAddress][i]                   \
                                [dcmPrivateCreatorAddress][0]               \
                                [dcmImageTypeAddress].value[2]
                if frameType    ==  "M_FFE":
                    self._magFrameIndex.append(i)   
                    
        #other manufacturers
        
        self._frames        = self._dcm.pixel_array[self._magFrameIndex]
        self._numFrames     = len(self._frames)
    
    
    
    ######################################################################
    #Functions dealing with the segmentation & interpolation of T1 & mask
    
    
    def findT1Grid(self):
        """
            Finds the locations of the voxels in the T1 image (and the mask)
            to be used in the interpolation. 
        """
        
        t1Spacing   = self._dcm[0x5200, 0x9230][0][0x0028, 0x9110]\
                                [0][0x0028, 0x0030].value
        t1Shape     = self._frames.shape
        self._x   = np.linspace(0,t1Shape[0]-1, t1Shape[0])
        self._y   = np.linspace(0,(t1Shape[2]-1)*t1Spacing[0], t1Shape[2])
        self._z   = np.linspace(0,(t1Shape[1]-1)*t1Spacing[1], t1Shape[1])
    
    def findInterpolatingGrid(self):
        """
            Finds the locations of the voxels in the PCA slice to be used in 
            the interpolation. 
        """
        pca        = self._pcaDcm.pixel_array
        pcaSlice   = np.reshape(pca[0], (1,pca.shape[1], pca.shape[2]))
        xi, yi, zi, pca_val = SELMAInterpolate.getInterpolationVariables(
                                            self._pcaDcm, pcaSlice)
        # zero coordinates
        #TODO: change for different image orientations
        r0         = SELMAInterpolate.usableImagePosition(self._dcm)
        self._xi  = xi+r0[0]
        self._yi  = yi-r0[1]
        self._zi  = -zi+r0[2]
        
        
        pts             = np.transpose((self._xi,self._zi,self._yi))
        self._idx       = (np.array(self._xi    >=  min(self._x)) * 
                           np.array(self._xi    <=  max(self._x)) * 
                           np.array(self._yi    >=  min(self._y)) * 
                           np.array(self._yi    <=  max(self._y)) *
                           np.array(self._zi    >=  min(self._z)) * 
                           np.array(self._zi    <=  max(self._z)) )
        self._pcaGrid   = pts[self._idx]
        
    def interpolateT1Slice(self):
        """
            Function that interpolates the correct T1 slice to match with 
            the PCA slices.
        """
        
        #Create interpolating function
        interpolatingFunc   = RegularGridInterpolator(
                                            (self._x, self._y, self._z),
                                            self._frames)
        
        #Find all points that fall within the T1 image space
        
        #Interpolate
        interpolated        = interpolatingFunc(self._pcaGrid)
        t1Slice             = np.zeros(self._idx.shape)
        t1Slice[self._idx]  = interpolated
        pca                 = self._pcaDcm.pixel_array
        self._t1Slice       = np.reshape(t1Slice, (pca.shape[-1], -1))
        
    def interpolateMaskSlice(self):
        """
            Function that interpolates the correct mask slice to match with 
            the PCA slices.
        """
        
        #Create interpolating function
        interpolatingFunc   = RegularGridInterpolator(
                                            (self._x, self._y, self._z),
                                            self._segmentation)
        
        #Find all points that fall within the T1 image space
        
        #Interpolate
        interpolated            = interpolatingFunc(self._pcaGrid)
        maskSlice               = np.zeros(self._idx.shape)
        maskSlice[self._idx]    = interpolated
        pca                     = self._pcaDcm.pixel_array
        self._maskSlice         = np.reshape(maskSlice, (pca.shape[-1], -1))
        
        

#functions
#prepare t1
    #rescale
    #align
    #skull stripping / hand-drawn mask
    #other preprocessing

#apply segmentation
#send segmentation back to selmadata
