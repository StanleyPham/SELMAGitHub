#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`SELMASegmentation`

"""

# ====================================================================

import SELMADicom
import SELMAInterpolate

import pydicom
import SimpleITK as sitk
import numpy as np
import matlab.engine


class SELMAT1Dicom(SELMADicom.SELMADicom):
    """
    This class deals with t1 segmentation
    """

    def __init__(self, dcmFilename, pcaDcm):
        """
        Load & rescale the T1 & interpolate correct slice
        """
        self._dcmFilename = dcmFilename
        self._dcm = pydicom.dcmread(self._dcmFilename)
        self._pcaDcm = pcaDcm

        #############################################################
        # Declare some variables for use in interpolating / segmenting

        # T1 properties
        self._manufacturer = None
        self._frames = None
        self._numFrames = None
        self._t1Slice = None
        self._magFrameIndex = None

        # interpolating properties
        self._M     = None      #transformation matrix between this T1 & pca

        # Segmentation properties
        self._maskSlice = None
        self._segmentation = None

        #############################################################

        # Get only the magnitude frames from the T1
        self.findManufacturer()
        self.findMagnitudeFrames()

        # Interpolate the t1 slice
        self.interpolateT1()

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
        
        if self._maskSlice is None:
            self.segmentAndInterpolateMask()
        
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
        self._manufacturer = self._dcm[0x8, 0x70].value

    def findMagnitudeFrames(self):
        """
            Finds the magnitude frames in the T1 dicom and stores the indices.        
        """

        self._magFrameIndex = []

        if self._manufacturer == "Philips Medical Systems":
            for i in range(len(self._dcm.pixel_array)):
                dcmFrameAddress = 0x5200, 0x9230
                dcmPrivateCreatorAddress = 0x2005, 0x140f
                dcmImageTypeAddress = 0x0008, 0x0008

                frameType = self._dcm[dcmFrameAddress][i] \
                    [dcmPrivateCreatorAddress][0] \
                    [dcmImageTypeAddress].value[2]
                if frameType == "M_FFE":
                    self._magFrameIndex.append(i)

                    # other manufacturers

        self._frames = self._dcm.pixel_array[self._magFrameIndex]
        self._numFrames = len(self._frames)

    ######################################################################
    # Functions dealing with the segmentation & interpolation of T1 & mask

    def interpolateT1(self):
        '''
        Interpolates a slice in the T1 image to match with the pca slice.
        '''
        
        #First, construct the 
        Mpca, Rpca  = SELMAInterpolate.getTransMatrix(self._pcaDcm)
        Mt1, Rt1    = SELMAInterpolate.getTransMatrix(self._dcm)
        self._M     = np.dot(np.linalg.inv(Mt1), Mpca)
        pcaShape    = self._pcaDcm.pixel_array.shape
        
        self._t1Slice   = SELMAInterpolate.doInterpolation(self._M,
                                                           self._frames,
                                                           pcaShape)
        
    def segmentAndInterpolateMask(self):
        '''
        Calls the matlab code that runs the SPM segmentation on the t1 dicom.
        Loads and interpolates the resulting WM mask.
        '''
        
        #Prepare for matlab call
        libraries = SELMAInterpolate.getLibraries()
        spm = libraries[0]
        dcm2nii = libraries[1]

        eng = matlab.engine.start_matlab()
        wm = eng.spmSegment(self._dcmFilename,
                             spm,
                             dcm2nii)

        # Load the WM segmentation
        im = sitk.ReadImage(wm)
        im = sitk.GetArrayFromImage(im)
        im = np.flip(im, 1)
        self._segmentation = im

        # Create interpolated slice
        pcaShape        = self._pcaDcm.pixel_array.shape
        self._maskSlice = SELMAInterpolate.doInterpolation(self._M,
                                                           self._segmentation,
                                                           pcaShape)


    ##### Old stuff. Can probably be removed.

    # def findT1Grid(self):
    #     """
    #         Finds the locations of the voxels in the T1 image (and the mask)
    #         to be used in the interpolation. 
            
    #         We switch from the T1 framework to the normal framework. in T1:
    #             z-axis increases from L->R
    #             y-axis increases from F->H (S->I)
    #             x-axis increases from A->P
                
    #         LPH
                
    #         Normally:
    #             z-axis increases from F->H (I->S)
    #             y-axis increases from A->P
    #             x-axis increases from R->L
                
    #         N.B this means that the z and y axes of the T1 framework also need
    #         to be inverted.
            
    #         Additionally, the data is structured (z,y,x)
    #     """

    #     t1Spacing = self._dcm[0x5200, 0x9230][0][0x0028, 0x9110] \
    #         [0][0x0028, 0x0030].value
    #     sliceThickness  = self._dcm[0x5200, 0x9230][0][0x0028, 0x9110] \
    #         [0][0x18, 0x50].value
            
    #         #Vervangen met spacing
            
    #     t1Shape = self._frames.shape
    #     self._x = np.linspace(0, (t1Shape[2] - 1) * t1Spacing[0], t1Shape[2])
    #     self._y = np.linspace(0, (t1Shape[1] - 1) * t1Spacing[1], t1Shape[1])
    #     self._z = np.linspace(0, (t1Shape[0] - 1) * sliceThickness, t1Shape[0])
        
    #     r0 = SELMAInterpolate.usableImagePosition(self._dcm)
        
    #     x_temp = r0[0] - self._z
    #     y_temp = self._x - r0[2] 
    #     z_temp = self._y + r0[1]
        
    #     self._x = x_temp
    #     self._y = y_temp
    #     self._z = z_temp
        
    #     #Invert the right dimensions:
    #     self._x = self._x[::-1]
    #     # self._z = self._z[::-1]
        
    #     self._frames = np.swapaxes(self._frames, 0,2)
    #     self._frames = np.swapaxes(self._frames, 0,1)
    #     # self._frames = np.swapaxes(self._frames, 1,2)
        
    #     self._frames = self._frames[:,:,::-1]


    # def findInterpolatingGrid(self):
    #     """
    #         Finds the locations of the voxels in the PCA slice to be used in 
    #         the interpolation. 
    #     """
    #     pca = self._pcaDcm.pixel_array
    #     pcaSlice = np.reshape(pca[0], (1, pca.shape[1], pca.shape[2]))
    #     self._xi, self._yi, self._zi, pca_val =     \
    #         SELMAInterpolate.getInterpolationVariables(
    #         self._pcaDcm, pcaSlice)
    #     # zero coordinates
    #     # TODO: change for different image orientations

    #     pts = np.transpose((self._zi, self._yi, self._xi))
    #     self._idx = (np.array(self._xi >= min(self._x)) *
    #                  np.array(self._xi <= max(self._x)) *
    #                  np.array(self._yi >= min(self._y)) *
    #                  np.array(self._yi <= max(self._y)) *
    #                  np.array(self._zi >= min(self._z)) *
    #                  np.array(self._zi <= max(self._z)))
    #     self._pcaGrid = pts[self._idx]

    # def interpolateT1Slice(self):
    #     """
    #         Function that interpolates the correct T1 slice to match with 
    #         the PCA slices.
    #     """
        
    #     #for testing only
    #     #make gradient
        
    #     # test =np.linspace(0, 5*np.max(self._frames), self._frames.shape[-1], 
    #     #                dtype=np.uint16)
    #     # self._frames *= test


    #     # Create interpolating function
    #     interpolatingFunc = RegularGridInterpolator(
    #         (self._z, self._y, self._x),
    #         self._frames)

    #     # Find all points that fall within the T1 image space

    #     # Interpolate
    #     interpolated = interpolatingFunc(self._pcaGrid)
    #     t1Slice = np.zeros(self._idx.shape)
    #     t1Slice[self._idx] = interpolated
    #     pca = self._pcaDcm.pixel_array
    #     self._t1Slice = np.reshape(t1Slice, (pca.shape[-1], -1))

    # def interpolateMaskSlice(self):
    #     """
    #         Function that interpolates the correct mask slice to match with 
    #         the PCA slices.
    #     """

    #     # Create interpolating function
    #     interpolatingFunc = RegularGridInterpolator(
    #         (self._z, self._y, self._x),
    #         self._segmentation)

    #     # Find all points that fall within the T1 image space

    #     # Interpolate
    #     interpolated = interpolatingFunc(self._pcaGrid)
    #     maskSlice = np.zeros(self._idx.shape)
    #     maskSlice[self._idx] = interpolated
    #     pca = self._pcaDcm.pixel_array
    #     self._maskSlice = np.reshape(maskSlice, (pca.shape[-1], -1))

