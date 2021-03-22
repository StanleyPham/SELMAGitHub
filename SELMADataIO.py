#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`SELMADataIO`

"""


# ====================================================================

import numpy as np

# ====================================================================
#IO

import pydicom
import imageio
import scipy.io
# import h5py
from PyQt5 import (QtCore, QtGui, QtWidgets)
# ====================================================================

import SELMAGUISettings

# ====================================================================

"""
This module contains all methods concerning IO.
"""

# def loadFlowDicom(fname):
#     """
#     Loads a single Dicom file containing the phase, modulus and
#     magnitude frames. 
    
#     Args:
#         fname(str): path to the Dicom file.
        
#     Returns:
#         dcm(pydicom.dcm): dicom object containing all the headers and 
#         pixel arrays.
#     """
    
#     dcm = pydicom.dcmread(fname)
#     return dcm

# def loadFlowDicomFromDirectory( fname):
#     """Loads all Dicom files in a directory.
    
#     Not implemented yet.
    
#     Args:
#          fname(str): path to directory containing Dicom.
            
#     Returns:
#         A merged Dicom object that behaves as if it's a non-classic Dicom.
#         """
#     pass

# # ====================================================================

# def loadDicom(fname):
#     """
#     Loads a single Dicom file containing the phase, modulus and
#     magnitude frames. 
    
#     Args:
#         fname(str): path to the Dicom file.
        
#     Returns:
#         dcm(pydicom.dcm): dicom object containing all the headers and 
#         pixel arrays.
#     """
#     pass

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

            #Non-h5 file
            maskDict    = scipy.io.loadmat(fname)
            maskKey     = list(maskDict.keys())[-1]
            mask        = maskDict[maskKey]
            
            
        except:
            # #H5 file, used for matlab v7.3 and higher
            # H5py currently doesn' work anymore, unsure why. 
            # User will be prompted to resave.
            # arrays = {}
            # f = h5py.File(fname, 'r')
            # for key, value in f.items():
            #     arrays[key] = np.array(value)
                
            # mask = arrays[key]
            # mask = np.swapaxes(mask, 0,1)
            return None
        
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
    ext         = fname[-4:]
  
    if      ext == ".png":
        mask = mask.astype(np.uint8)
        imageio.imwrite(fname, mask)
    elif    ext == ".npy":
        np.save(fname, mask)
        scipy.io.savemat(fname[0:len(fname)-4], {'WMslice': mask})
        
    
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

def writeVesselDict(vesselDict, addonDict, fname):
    """
    Writes the vesselDict object to a .txt file.
    
    Args:
        vesselDict(dict): dictionary containing all the analysis values
        of all the significant vessels in the analysed dicom.
        
        fname(str): path to where the dictionary needs to be saved.
    
    """
    
    #Find if the decimalComma setting is turned on
    COMPANY, APPNAME, _ = SELMAGUISettings.getInfo()
    COMPANY             = COMPANY.split()[0]
    APPNAME             = APPNAME.split()[0]
    settings            = QtCore.QSettings(COMPANY, APPNAME)
    decimalComma        = settings.value('decimalComma') == 'true'
    
    with open(fname, 'w') as f:    
        #Write headers
        for key in vesselDict[0].keys():
            f.write(key)
            f.write('\t')
        f.write('\n')
    
        #Write vesseldata
        for i in range(len(vesselDict))    :
            for key in vesselDict[0].keys():
                text    = str(vesselDict[i][key])
                if decimalComma:
                    text    = text.replace('.',',')
                f.write(text)
                f.write('\t')
            f.write('\n')
            
        #Write additional info
        f.write('\n')
        for key in addonDict.keys():
            f.write(key)
            f.write('\t')
            f.write(str(addonDict[key]))
            f.write('\n')
            
def writeVelocityDict(velocityDict, addonDict, fname):
    """
    Writes the velocityDict object to a .txt file. This is a separate text file
    from the vesselDict object
    
    Args:
        velocityDict(dict): dictionary containing all the average velocities
        of all the significant vessels in the analysed dicom.
        
        fname(str): path to where the dictionary needs to be saved.
    
    """
    
    #Find if the decimalComma setting is turned on
    COMPANY, APPNAME, _ = SELMAGUISettings.getInfo()
    COMPANY             = COMPANY.split()[0]
    APPNAME             = APPNAME.split()[0]
    settings            = QtCore.QSettings(COMPANY, APPNAME)
    decimalComma        = settings.value('decimalComma') == 'true'

    with open(fname, 'w') as f:    

        for key in velocityDict[0].keys():
            f.write(key)
            f.write('\t')
            text    = str(velocityDict[0][key])
            if decimalComma:
                text    = text.replace('.',',')
            f.write(text)
            f.write('\n')
            
        #Write additional info
        f.write('\n')
        for key in addonDict.keys():
            f.write(key)
            f.write('\t')
            f.write(str(addonDict[key]))
            f.write('\n')
            

    
    
    
    
    