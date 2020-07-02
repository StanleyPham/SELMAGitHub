#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`SELMASegmentation`

"""


# ====================================================================

import SELMADicom

#import nipy.algorithms.segmentation as seg
import pydicom
#from deepbrain import Extractor
#from scipy.ndimage.morphology import binary_closing


class SELMAT1Dicom(SELMADicom.SELMADicom):
    """
    This class deals with t1 segmentation
    """
    
    def __init__(self, dcmFilename):
        """
        Load & rescale the T1 & interpolate correct slice
        """
        self._dcmFilename   = dcmFilename
        self._DCM           = pydicom.dcmread(self._dcmFilename)
        
        self._tags          = dict()
        self._rawFrames     = self._DCM.pixel_array
        self._numFrames     = len(self._rawFrames)
        
        #T1 properties
        self._T1Slice       = None
        
        # Segmentation properties
        self._brainMask         = None
        self._wmSegmentation    = None
        
        
        
        #### init main dicom values
        #Get manufacturer
        self._findManufacturer()
        
        #find important Tags
        self._findRescaleValues()    
        self._findVEncoding()
        self._findFrameTypes()
        
        self._findTargets()
        
        #Get rescale values and apply
        self._rescaleFrames()
    
    
    '''Public'''
    def getSegmentationMask(self, pcaDicom):
        """
        Walks through the various functions for constructing a T1 segmentation.
        
        Notably:
            -Align t1 with 2d pca slice (find orientation)
            -Get brain mask
            -...
            -perform segmentation
        """
        pass
    
    
    
    '''Private'''
    
    def interpolateT1Slice(self):
        #Get dcm header for pca
        #Get dcm header for t1
        #Create orientation vector
        #Create coordinate matrix
        #Interpolate 
        pass
    


#functions
#prepare t1
    #rescale
    #align
    #skull stripping / hand-drawn mask
    #other preprocessing

#apply segmentation
#send segmentation back to selmadata
