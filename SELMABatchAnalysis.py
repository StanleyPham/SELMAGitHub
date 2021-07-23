# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:10:14 2021

@author: spham2
"""

import os
import numpy as np
from PyQt5 import (QtCore)

import SELMAData
import SELMADataIO
import SELMADataModels

def EnhancedBatchAnalysis(dirName, files, self):

    #Make list of all suitable .dcm files
    dcms = []
    for file in files:
        if file.find(".dcm") != -1 and file.find("mask") == -1:
            dcms.append(file)
  
    i       = 0 
    total   = len(dcms)
    
    if not dcms:
    
        self.signalObject.errorMessageSignal.emit(
             "No DICOM files found in folder. This batch job will "+
             "be stopped.")
        
        return

    batchAnalysisResults = dict()
    
    #Iterate over all suitable .dcm files.
    for dcm in dcms:

        self.signalObject.setProgressLabelSignal.emit(
                "Patient %.0f out of %.0f" %(i + 1, total))
        
        self._SDO   = SELMAData.SELMADataObject(self.signalObject,
                                                dcmFilename = dirName 
                                                + '/' + dcm,
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
                        
                        mask = SELMADataIO.loadMask(dirName + '/' + 
                                                    file)
                                                
                        self._SDO.setMask(mask)
                        
                    except:
                    
                        self.signalObject.errorMessageSignal.emit(
            "The mask of %s has a version of .mat file that " %(dcm) +
            "is not supported. Please save it as a non-v7.3 file "+
            "and try again. Moving on to next scan.")
                        
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
                        
        if not bool(vesselDict):
            
            continue
        
        addonDict = SELMADataIO.getAddonDict(self)
        
        outputName = dirName + '/' + name + "-Vessel_Data.txt"
        SELMADataIO.writeVesselDict(vesselDict, addonDict, outputName)
        outputName = (dirName + '/' + name + "-averagePIandVelocity"
        + "_Data.txt")
        SELMADataIO.writeVelocityDict(velocityDict, addonDict, 
                                      outputName)
  
        #Save in single file
        #batchAnalysisResults[i] = self._SDO.getBatchAnalysisResults()
        batchAnalysisResults[i] = SELMADataIO.getBatchAnalysisResults(self._SDO)
        
        outputName = dirName + '/batchAnalysisResults.mat' 
        SELMADataIO.writeBatchAnalysisDict(batchAnalysisResults, 
                                           outputName)
              
        #Emit progress to progressbar
        self.signalObject.setProgressBarSignal.emit(int(100 * i / 
                                                        total))
            
        i += 1
    
    outputName = dirName + '/batchAnalysisResults.mat' 
    SELMADataIO.writeBatchAnalysisDict(batchAnalysisResults, 
                                       outputName)
    
    #Emit progress to progressbar
    self.signalObject.setProgressBarSignal.emit(int(100))
    self.signalObject.setProgressLabelSignal.emit(
                "Batch analysis complete!"
                )