#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`SELMADataIO`

"""


# ====================================================================

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#from future_builtins import *

# ====================================================================

import numpy as np
from PyQt5 import (QtCore, QtGui, QtWidgets)


# ====================================================================
#IO

import SimpleITK as sitk
import pydicom
import imageio
import scipy.io
import h5py
# ====================================================================

# ====================================================================

"""
This module contains all methods concerning IO.
"""

def loadFlowDicom(fname):
    """
    Loads a single Dicom file containing the phase, modulus and
    magnitude frames. 
    
    Args:
        fname(str): path to the Dicom file.
        
    Returns:
        dcm(pydicom.dcm): dicom object containing all the headers and 
        pixel arrays.
    """
    
    dcm = pydicom.dcmread(fname)
    return dcm

def loadFlowDicomFromDirectory( fname):
    """Loads all Dicom files in a directory.
    
    Not implemented yet.
    
    Args:
         fname(str): path to directory containing Dicom.
            
    Returns:
        A merged Dicom object that behaves as if it's a non-classic Dicom.
        """
    pass

# ====================================================================

def loadDicom(fname):
    """
    Loads a single Dicom file containing the phase, modulus and
    magnitude frames. 
    
    Args:
        fname(str): path to the Dicom file.
        
    Returns:
        dcm(pydicom.dcm): dicom object containing all the headers and 
        pixel arrays.
    """
    pass

def loadMask( fname):
    """Loads a mask file. The following types are supported:
        .png
        .npy
        .mat
        
    Args:
        fname(str): path to the mask file.
        
    Returns:
        numpy.ndarray containing the binary mask. 
    """
    
    #find extension
    ext = fname[-4:]
    
    if      ext == ".png":
        mask = imageio.imread(fname)
    elif    ext == ".npy":
        mask = np.load(fname)
    elif    ext == ".mat":
        
        try:
            #H5 file
            arrays = {}
            f = h5py.File(fname, 'r')
            for key, value in f.items():
                arrays[key] = np.array(value)
                
            mask = arrays[key]
            mask = np.swapaxes(mask, 0,1)
        except:
            #Non-h5 file
            maskDict    = scipy.io.loadmat(fname)
            maskKey     = list(maskDict.keys())[-1]
            mask        = maskDict[maskKey]
        
    return mask


def saveMask(fname, mask):
    """Saves a mask file. The following types are supported:
        .png
        .npy
        
    Args:
        fname(str): path to where the mask is saved.
        mask(numpy.ndarray): the mask to be saved.
    """

    #find extension
    fname       = fname[0]
    ext         = fname[-4:]
    
    if      ext == ".png":
        mask = mask.astype(np.uint8)
        imageio.imwrite(fname, mask)
    elif    ext == ".npy":
        np.save(fname, mask)
        
        
    
#    elif    ext == ".mat":
#        mask = makeMatlabDict(mask)
#        
#        
#def makeMatlabDict(mask):
#    """
#    Turns an ndarray object into a dictionary that can be read by
#    Matlab.
#    
#    Not implemented yet.
#    
#    Args:
#        mask(numpy.ndarray): binary mask
#    
#    Returns
#        maskDict(dict): dictionary that can be saved to .mat file.
#    
#    """
#    
#    #TODO: implement
#    
#    return mask

def writeVesselDict(vesselDict, fname):
    """
    Writes the vesselDict object to a .txt file.
    
    Args:
        vesselDict(dict): dictionary containing all the analysis values
        of all the significant vessels in the analysed dicom.
        
        fname(str): path to where the dictionary needs to be saved.
    
    """
    
    with open(fname, 'w') as f:    
        for key in vesselDict[0].keys():
            f.write(key)
            f.write('\t')
        f.write('\n')
    
        for i in range(len(vesselDict))    :
            for key in vesselDict[0].keys():
                f.write(str(vesselDict[i][key]))
                f.write('\t')
            f.write('\n')


    
    
    
    
    