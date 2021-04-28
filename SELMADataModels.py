#!/usr/bin/env python

"""
This module is contains all the relevant classes that form the second layer 
between the SELMA GUI and the data objects. It contains the following classes:

+ :class: `SDMSignals`
+ :class: `SelmaDataModel`
    
"""

# ====================================================================

import numpy as np
from PyQt5 import (QtCore)
import os
import scipy

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
    setProgressLabelSignal  = QtCore.pyqtSignal(str) 
    setFrameCountSignal     = QtCore.pyqtSignal(int, int) 
    pixelValueSignal        = QtCore.pyqtSignal(int, int, float)
    errorMessageSignal      = QtCore.pyqtSignal(str)
    infoMessageSignal       = QtCore.pyqtSignal(str)
    
    sendImVarSignal         = QtCore.pyqtSignal(dict)
    
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
        
        self._displayT1     = False;
        
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
        
        if not self._displayT1:
            self._frameCount += direction
            if self._frameCount <= 0:
                self._frameCount = self._frameMax
                
            if self._frameCount > self._frameMax:
                self._frameCount = 1
            
        self._displayFrame()
    
    
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
        
        mask = SELMADataIO.loadMask(fname)
        if mask is None:
            self.signalObject.errorMessageSignal.emit(
                "This version of .mat file is not supported. Please " +
                "save it as a non-v7.3 file and try again.")
            return
        
        
        #Ensure that the mask has the same dimensions as the Frames
        frames = self._SDO.getFrames()
        frame = frames[self._frameCount - 1]
        maskShape   = mask.shape
        frameShape  = frame.shape
        
        if maskShape != frameShape:
            errStr  = "The dimensions of the frame and the mask do not align. "
            self.signalObject.errorMessageSignal.emit(errStr + 
                                                      str(frameShape) + 
                                                      str(maskShape))
            
        else:
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
    
    def segmentMaskSlot(self):
        """
        
        """
        
        if self._SDO is None:
            self.signalObject.errorMessageSignal.emit(
                    "Please load a PCA dicom first.")
            return
        if self._SDO.getT1() is None:
            self.signalObject.errorMessageSignal.emit(
                    "Please load a T1 dicom first.")
            return
        
        self._SDO.segmentMask()
        
        mask = self._SDO.getMask()
        print(mask.shape, np.unique(mask))
        self.signalObject.sendMaskSignal.emit(mask)
        
    def thresholdMaskSlot(self):
        """Gets a new copy of the (thresholded) mask from the SDO and 
        returns it to the GUI"""
        
        if self._SDO is None:
            return
        
        mask = self._SDO.getMask()
        if mask is not None:
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
        
        self._SDO   = SELMAData.SELMADataObject(self.signalObject,
                                                dcmFilename= fname)
        self._frameCount    = 1
        self._frameMax      = self._SDO.getNumFrames()
        
        self._displayFrame()
            
    def loadClassicDCMSlot(self, fnames):
        """
        Loads a new classic DCM into the SDO. Triggered when the 
        openClassicAct is called.
        
        Args:
            fnames(tuple(str)): list of filenames
                
        """
        if fnames is None:
            return
 
        self._SDO   = SELMAData.SELMADataObject(self.signalObject,
                                                dcmFilename=fnames,
                                                classic = True)
        self._frameCount = 1
        self._frameMax = self._SDO.getNumFrames()
        
        self._displayFrame()
    
    def loadT1DCMSlot(self, fname):
        """
        Loads a new T1 DCM into the program. Triggered when the 
        openT1Act is called.
        
        Args:
            fname (str): path to the Dicom file.
                
        """
        if fname is None:
            return
        if self._SDO is None:
            self.signalObject.errorMessageSignal.emit(
                    "Please load a PCA dicom first.")
            return
        
        self._SDO.setT1(fname)
        self._t1FrameCount  = 1
        self._t1FrameMax    = self._SDO.getT1().getNumFrames()
        self._displayT1     = True
        
        self._displayFrame()
        
    
    
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

        self.signalObject.infoMessageSignal.emit(
                 "GUI may become unresponsive while executing batch analysis. "+
                 "Please do not close GUI until batch analysis is complete "+
                 "or an error has occured. Press OK to continue.")
        
        # os.chdir(dirName)
    
        files = os.listdir(dirName)
        
        if not any(os.path.isdir(dirName + '/' + subfolder) for subfolder in files):
     
            #Make list of all suitable .dcm files
            dcms = []
            for file in files:
                if file.find(".dcm") != -1 and file.find("mask") == -1:
                    dcms.append(file)
      
            i       = 0 
            total   = len(dcms)
            
            if not dcms:
            
                self.signalObject.errorMessageSignal.emit(
                     "No DICOM files found in folder. This batch job will be "+
                     "stopped.")
                
                return
    
            batchAnalysisResults = dict()
            
            #Iterate over all suitable .dcm files.
            for dcm in dcms:
    
                self.signalObject.setProgressLabelSignal.emit(
                        "Patient %.0f out of %.0f" %(i + 1, total))
                
                self._SDO   = SELMAData.SELMADataObject(self.signalObject,
                                                        dcmFilename= dirName + '/' + dcm,
                                                        classic = False)
                
                name        = dcm[:-4]
                #find mask
    
                for file in files:
                    if file.find(name) != -1 and file.find("mask") != -1:
                        if file[-4:] == ".dcm" or file[-4:] == ".npy":
                            #Now it will find the dcm object itself.
    #                        self._SDO.setT1(file)
    #                        break
                            pass
                        else:
                            
                            try:
                                
                                mask = SELMADataIO.loadMask(dirName + '/' + file)
                                                        
                                self._SDO.setMask(mask)
                                
                            except:
                            
                                self.signalObject.errorMessageSignal.emit(
                    "The mask of %s has a version of .mat file that " %(dcm) +
                    "is not supported. Please save it as a non-v7.3 file and "+
                    "try again. Moving on to next scan.")
                                
                                #return
    
                                break
                
                #If no mask is found, move on to the next image
                if self._SDO.getMask() is None:
                    
                    self.signalObject.infoMessageSignal.emit(
                     "Mask of %s not found in folder. Moving to next scan"
                     %(dcm))
                    
                    continue
                
                #Do vessel analysis
                self._SDO.analyseVessels()
                
                #Save results
                #TODO: support for other output types.
                vesselDict, velocityDict = self._SDO.getVesselDict()
                addonDict = self._SDO.getAddonDict()
                
                outputName = dirName + '/' + name + "-Vessel_Data.txt"
                SELMADataIO.writeVesselDict(vesselDict, addonDict, outputName)
                outputName = dirName + '/' + name + "-averagePIandVelocity_Data.txt"
                SELMADataIO.writeVelocityDict(velocityDict, addonDict, outputName)
    
                #Save in single file
                batchAnalysisResults[i] = self._SDO.getBatchAnalysisResults()
                
                outputName = dirName + '/batchAnalysisResults.mat' 
                SELMADataIO.writeBatchAnalysisDict(batchAnalysisResults, outputName)
                      
                #Emit progress to progressbar
                self.signalObject.setProgressBarSignal.emit(int(100 * i / total))
                    
                i += 1
            
            outputName = dirName + '/batchAnalysisResults.mat' 
            SELMADataIO.writeBatchAnalysisDict(batchAnalysisResults, outputName)
            
            #Emit progress to progressbar
            self.signalObject.setProgressBarSignal.emit(int(100))
            self.signalObject.setProgressLabelSignal.emit(
                        "Batch analysis complete!"
                        )
            
        elif any(os.path.isdir(dirName + '/' + subfolder) for subfolder in files):
          
            i       = 0 
            total   = len(files)
            
            batchAnalysisResults = dict()

            for subject in files:
                
                if not os.path.isdir(dirName + '/' + subject):
                    
                    continue
                
                subject_folder = os.listdir(dirName + '/' + subject)
                
                dcmFilename = []
                
                for file in subject_folder:
                    
                    if file.endswith('.txt'):
                        
                        continue
                    
                    elif file.endswith('.mat'):
                        
                        if file.find('mask') != -1:
                            
                            try:
                                
                                mask = SELMADataIO.loadMask(dirName + '/' + subject + '/' + file)
            
                            except:
                            
                                self.signalObject.errorMessageSignal.emit(
                    "The mask of %s has a version of .mat file that " %(subject) +
                    "is not supported. Please save it as a non-v7.3 file and "+
                    "try again. Moving on to next scan.")
                                
                                #return
    
                                break
                        
                        continue
                    
                    elif file.endswith('.log'):
                        
                        continue
                    
                    elif file.endswith('.dcm'):
                        
                        continue
                    
                    elif file.endswith('.npy'):
                        
                        continue
                    
                    elif os.path.getsize(dirName + '/' + subject + '/' + file) < 100000:
                        
                        continue
                    
                    dcmFilename.append(dirName + '/' + subject + '/' + file)
                    
                if dcmFilename == []:
                    
                    continue
                
                self.signalObject.setProgressLabelSignal.emit(
                        "Patient %.0f out of %.0f" %(i + 1, total))
                
                self._SDO   = SELMAData.SELMADataObject(self.signalObject,
                                                        dcmFilename ,
                                                        classic = True)
                
                self._SDO.setMask(mask)
                                         
                #If no mask is found, move on to the next image
                if self._SDO.getMask() is None:
                    
                    self.signalObject.infoMessageSignal.emit(
                     "Mask of %s not found in folder. Moving to next subject"
                     %(subject))
                    
                    continue
                
                #Do vessel analysis
                self._SDO.analyseVessels()
                
                #Save results
                #TODO: support for other output types.
                vesselDict, velocityDict = self._SDO.getVesselDict()
                addonDict = self._SDO.getAddonDict()
                
                outputName = dirName + '/' + subject + "-Vessel_Data.txt"
                SELMADataIO.writeVesselDict(vesselDict, addonDict, outputName)
                outputName = dirName + '/' + subject + "-averagePIandVelocity_Data.txt"
                SELMADataIO.writeVelocityDict(velocityDict, addonDict, outputName)
    
                #Save in single file
                batchAnalysisResults[i] = self._SDO.getBatchAnalysisResults()
                
                outputName = dirName + '/batchAnalysisResults.mat' 
                SELMADataIO.writeBatchAnalysisDict(batchAnalysisResults, outputName)
                      
                #Emit progress to progressbar
                self.signalObject.setProgressBarSignal.emit(int(100 * i / total))
                    
                i += 1
                
            outputName = dirName + '/batchAnalysisResults.mat' 
            SELMADataIO.writeBatchAnalysisDict(batchAnalysisResults, outputName)
            
            #Emit progress to progressbar
            self.signalObject.setProgressBarSignal.emit(int(100))
            self.signalObject.setProgressLabelSignal.emit(
                        "Batch analysis complete!"
                        )
                                   
    def switchViewSlot(self):
        if self._SDO is None:
            return
        if self._SDO.getT1() is None:
            return
            
        self._displayT1 = not self._displayT1
        self._displayFrame()
        
        
        
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
        
    def getVarSlot(self):
        
        if self._SDO is None:
            return
        
        variables = dict()
        
        venc                    = self._SDO.getVenc()
        variables['venc']       = venc
        
        velRescale              = self._SDO.getRescale()
        variables['velscale']   = velRescale
        
        #Return the variables
        self.signalObject.sendImVarSignal.emit(variables)
    
    def setVarSlot(self, variables):
        """Sets the user-defined variables stored in the ImVar window"""
        
        if self._SDO is None:
            self.signalObject.errorMessageSignal.emit("No DICOM loaded.")
            return
            
        for variable in variables:
            
            if variable == "venc":
                self._SDO.setVenc(variables['venc'])
            
            if variable == "velscale":
                self._SDO.setVelRescale(variables["velscale"])
                
            #other variables
        
        
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
        

    def _displayFrame(self):
        
        
        if self._displayT1:
            frame   = self._SDO.getT1().getFrames()
#            frame   = frames[self._t1FrameCount - 1]
            self.signalObject.setFrameCountSignal.emit(1, 1)
            
        else:
            frames      = self._SDO.getFrames()
            frame       = frames[self._frameCount - 1]
            self.signalObject.setFrameCountSignal.emit(self._frameCount,
                                                       self._frameMax)
            
        self.signalObject.setPixmapSignal.emit(frame)
        
        

