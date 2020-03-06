#!/usr/bin/env python

"""
This module is contains all the relevant classes that form the second layer 
between the SELMA GUI and the data objects. It contains the following classes:

+ :class: `SDMSignals`
+ :class: `SelmaDataModel`
    
"""

# ====================================================================

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#from future_builtins import *

# ====================================================================

import numpy as np
from PyQt5 import (QtCore)
import os

# ====================================================================

import SELMAData
import SELMADataIO

# ====================================================================


class SDMSignals(QtCore.QObject):
    """
    This class inherits from a QObject in order to store and connect
    pyqtSignals.
    
    """
    
    setPixmapSignal         = QtCore.pyqtSignal(np.ndarray) 
    sendVesselMaskSignal    = QtCore.pyqtSignal(np.ndarray) 
    sendMaskSignal          = QtCore.pyqtSignal(np.ndarray) 
    setProgressBarSignal    = QtCore.pyqtSignal(int) 
    setFrameCountSignal     = QtCore.pyqtSignal(int, int) 
    pixelValueSignal        = QtCore.pyqtSignal(int, int, float)
    errorMessageSignal      = QtCore.pyqtSignal(str)
    
class SelmaDataModel:
    """
    This class is the hub that manages all the data that is used in the 
    program. It contains all the slots for signals sent from the GUI that
    interact with the data in some way.
    
    The class contains an instance of a SELMAData object, as well as all
    variables needed to perform the vessel analysis. These are read from the
    user settings.
    
    The class furthermore handles sending data to the GUI as well as calling
    IO methods from SELMADataIO.
    """
    
    
    def __init__(self, selmaDataObject = None):

        self._SDO = selmaDataObject        
#        self._medianDiameter = medianDiameter
        
        self._frameCount = 1
        self._frameMax   = 0
        
        self.signalObject = SDMSignals()
        
        
    '''Public'''
     
    #Slots
    # ------------------------------------------------------------------    
    
    def newFrameSlot(self, direction):
        """
        Triggered when an unmodified wheelEvent happens in
        SELMAGUI.        
        Cycles through the stored frames based on the direction of the
        mouseWheel. Sends signals to set the frameLabel and the pixmap.
        
        Args:
            direction (int): +1 or -1, the direction of the scrollEvent.
            
        Returns:
            Sends signals with the following:
                frame (numpy.ndarray): the frame at the new frameCount.
                
                frameCount (int): the new frameCount. 
                frameMax   (int): the total number of Frames.
        """
        
        if self._SDO is None:
            return
        
        self._frameCount += direction
        if self._frameCount <= 0:
            self._frameCount = self._frameMax
            
        if self._frameCount > self._frameMax:
            self._frameCount = 1
            
        frames = self._SDO.getFrames()
        frame = frames[self._frameCount - 1]
        self.signalObject.setPixmapSignal.emit(frame)
        
        self.signalObject.setFrameCountSignal.emit(self._frameCount,
                                                   self._frameMax)
    
    
    def loadMaskSlot(self, fname):
        """
        Calls the loadMask function from SELMADataIO and sends the loaded
        mask back to the GUI.
        
        Args:
            fname (str): path to the mask.
            
        Returns:
            Signal with the following:
                mask (numpy.ndarray): the mask which was referred to.        
        """
        
        if fname is None or self._SDO is None:
            return
        
        mask = SELMADataIO.loadMask(fname[0])
        
        self._SDO.setMask(mask)
        self.signalObject.sendMaskSignal.emit(mask)
        
    def saveMaskSlot(self, fname):
        """
        Gets the mask (if any) from the SDO and calls the saveMask function in 
        SELMADataIO.
        
        Args:
            fname (str): path to where the mask needs to be saved.
            
        Returns:
        """
        
        mask = self._SDO.getMask()
        SELMADataIO.saveMask(fname, mask)
    
    def segmentMaskSlot(self, fname):
        """
        Calls the loadT1 function from SELMADataIO and sends the SELMADicom
        object to the SDO to be segmented. Then sends the mask back to the GUI.
        
        Args:
            fname (str): path to the T1 that needs to be segmented.
            
        Returns:
            Signal with the following:
                mask (numpy.ndarray): the segmented mask.        
        """
        
        if fname is None or self._SDO is None:
            return
        
        self._SDO.setT1(fname[0])
        
        mask = self._SDO.getMask()
        self.signalObject.sendMaskSignal.emit(mask)
        

    def loadDCMSlot(self, fname):
        """
        Loads a new DCM into the SDO. Triggered when the openAct is called.
        
        Args:
            fname (str): path to the Dicom file.
            
        Returns:
            Sends signals with the following:
                frame (numpy.ndarray): the frame at the current frameCount.
                
                frameCount (int): the current frameCount. 
                frameMax   (int): the total number of Frames.
        
        """
        
        if fname is None:
            return
        
        self._SDO   = SELMAData.SELMADataObject(fname[0],
                                                self.signalObject)
        frames      = self._SDO.getFrames()
        frame       = frames[0]
        self.signalObject.setPixmapSignal.emit(frame)
        
        #update frameLabel
        self._frameMax = len(frames)
        self.signalObject.setFrameCountSignal.emit(self._frameCount,
                                                   self._frameMax)
            
    def loadDCMDirSlot(self, fname):
        """
        Loads a new classic DCM into the SDO. Triggered when the 
        openDicomDirAct is called.
        
        Not implemented yet.
        
        Args:
            fname(str): path to directory containing Dicom.
                
        """
        pass
    
    def applyMaskSlot(self, mask):
        """
        Sets the drawn mask into the data object.
        
        Args:
            mask (numpy.ndarray): mask from the GUI.
        
        """
        self._SDO.setMask(mask)
    
    
    def analyseVesselSlot(self):
        """
        Slot for analyseVesselSignal. Tells the SDO to analyse the vessels
        in its current dataset.
        
        """
        if self._SDO is None:
            self.signalObject.errorMessageSignal.emit("No DICOM loaded.")
            return
        
        self._SDO.analyseVessels()
        
    
    def analyseBatchSlot(self, dirName):
        """Slot for the analyse batch signal.
        Goes through the specified directory and finds all .dcm files which
        do not have 'mask' in the name. The program then iterates over these
        files:
            A SelmaDataObject is created with the .dcm file.
            The directory is then searched for a mask file which has the same
            name as the .dcm (along with 'mask' somewhere in the name).
            This can be any suitable mask type, or another Dicom file, in which
            case a segmentation is made.
            Then the vesselAnalysis function is called and the results are
            written to a .txt file with the same name.
            
        Args:
            dirname(str): path to the directory containing all input files.
        """
        
        #TODO: add progress feedback
        
        os.chdir(dirName)
        files = os.listdir()
        
        #Make list of all suitable .dcm files
        dcms = []
        for file in files:
            if file.find(".dcm") != -1 and file.find("mask") == -1:
                dcms.append(file)
        
        
        #Iterate over all suitable .dcm files.
        for dcm in dcms:
            
            self._SDO   = SELMAData.SELMADataObject(dcm)
            
            name        = dcm[:-4]
            #find mask
            for file in files:
                if file.find(name) != -1 and file.find("mask") != -1:
                    if file[-4:] == ".dcm":
                        self._SDO.setT1(file)
                    else:
                        mask = SELMADataIO.loadMask(file)
                        self._SDO.setMask(mask)
            
            #Do vessel analysis
            self._SDO.analyseVessels()
            
            #Save results
            #TODO: support for other output types.
            outputName = name + ".txt"
            vesselDict = self._SDO.getVesselDict()
            SELMADataIO.writeVesselDict(vesselDict, outputName)
        
        
    def saveVesselStatisticsSlot(self, fname):
        """
        Slot for saveVesselStatisticsSignal. Saves the statistics of the 
        significant vessels to the filename.
        
        Args:
            fname (str): path to where the result of the analysis should be
            written.
        """
        
        vesselDict = self._SDO.getVesselDict()
        SELMADataIO.writeVesselDict(vesselDict, fname)
        
    def pixelValueSlot(self, x,y):
        """
        Slot for mouseMoveEvent in the GUI. Sends back the cursor location
        as well as the value of the current frame under that location.
        
        Args:
            x (int): x-index of the frame
            y (int): y-index of the frame
            
        Returns:
            Sends the following via a signal:
                x (int): x-index of the frame
                y (int): y-index of the frame
                pixelValue (float): value of the current frame at [x,y]
        """        
        
        if self._SDO is None:
            return
        
        frames      = self._SDO.getFrames()
        frame       = frames[self._frameCount - 1]
        pixelValue  = frame[y,x]
        
        self.signalObject.pixelValueSignal.emit(x, y, pixelValue)
        
    #Getter functions
    # ------------------------------------------------------------------    
    
    def getSDO(self):
        return self._SDO
    
#    def getMedianDiameter(self):
#        return self._medianDiameter
    
    #Setter functions
    # -----------------------------------------------------------------
    
#    def setSDO(self, selmaDataObject):
#        self._SDO = selmaDataObject
        
#    def setMedianDiameter(self, diam):
#        self._medianDiameter = diam
    
        
    
    '''Private'''
        
