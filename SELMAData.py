#!/usr/bin/env python

"""
This module is contains all the relevant classes that form the data layer 
of the SELMA project. It contains the following classes:

+ :class:`SELMADataObject`
    
"""

# ====================================================================

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#from future_builtins import *

# ====================================================================

import numpy as np
import scipy.signal
import scipy.stats
#from dipy.segment import tissue
#from dipy import io
import cv2
from multiprocessing import Pool, freeze_support, cpu_count

from PyQt5 import QtCore

# ====================================================================

import SELMADicom
import SELMADataIO

# ====================================================================

# -------------------------------------------------------------
'''Auxillary functions, used in the vessel analysis'''

def div0(a, b ):
    """ Divide function that ignores division by 0:
        div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    #from: https://stackoverflow.com/questions/26248654/
    #   how-to-return-0-with-divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c


def applyMedianFilter(obj):
    """Performs a median filter on the array with the specified diameter"""
    diameter, array = obj
    return scipy.signal.medfilt2d(array, diameter)



class SELMADataObject:
    """This class stores all data used in the program. It has a SELMADicom
    object for the Dicom image of the flow data, as well as the T1.
    Any segmentation and vessel masks are also stored here.
    
    The class furthermore contains all the methods for analysing and 
    directly handling the data. It is called by SELMADataModels, which 
    manages the user specified settings."""
    
    def __init__(self, dcmFilename = None, signalObject = None):
        
        self._mask          = None
        self._t1            = None
        self._vesselMask    = None
        self._selmaDicom    = None
        
        if dcmFilename is not None:
            self._selmaDicom    = SELMADicom.SELMADicom(dcmFilename)
            self._dcmFilename   = dcmFilename
            
        self._signalObject = signalObject
    
    '''Public'''
    
    # 
    # ------------------------------------------------------------------
    
    def analyseVessels(self):
        #TODO: apply settings
        
        if self._selmaDicom is None:
            self._signalObject.errorMessageSignal.emit("No DICOM loaded.")
            return
        
        self._calculateMedians()
        self._subtractMedian()
        
        #Determine SNR of all voxels
        self._SNR()
        
        #Find all vessels with significant flow.
        self._findSignificantFlow()
        
        self._removeZeroCrossings()
        self._applyT1Mask()
        
        
        self._findSignificantMagnitude()
        self._clusterVessels()
        
        #Send vessels back to the GUI
        self._signalObject.sendVesselMaskSignal.emit(self._vesselMask)
        
        #make dictionary and write to disk
        self._makeVesselDict()
        self._writeToFile()
        
    
    
    #Getter functions
    # ------------------------------------------------------------------    
    def getFrames(self):
        return self._selmaDicom.getFrames()
    
    def getRawFrames(self):
        return self._selmaDicom.getRawFrames()

    def getMask(self):
        return self._mask
    
    def getT1(self):
        return self._t1
    
    def getVesselMask(self):
        return self._vesselMask
#    
    def getVesselDict(self):
        return self._vesselDict
    
    def getDcmFilename(self):
        return self._dcmFilename
#    
    
    #Setter functions
    # ------------------------------------------------------------------    
    
    def setMask(self, mask):
        self._mask = mask
        
    def setT1(self, t1Fname):
        self._t1 = SELMADicom.SELMADicom(t1Fname)
        self._segmentT1()
        
        
    '''Private'''
    # Setup data from .dcm file
    # ------------------------------------------------------------------    
    
    
    def readFromSettings(self, key):
        """Loads the settings object associated with the program and 
        returns the value at the key."""
        
        #TODO, don't hardcode this
        COMPANY = "UMCu"
        APPNAME = "SELMA"
        
        settings = QtCore.QSettings(COMPANY, APPNAME)
        val = None
        
        try:
            val = settings.value(key)
        except:
            self._signalObject.errorMessageSignal.emit("Wrong setting accessed.")
            return val
        
        #Return the right type
        if val == "true":
            return True
        if val == "false":
            return False
        
        return float(val)
    
    def getSigma(self):
        """ Returns the upper end of the confidence interval with the alpha
        value in the settings.
        
        Args:
            
        Returns:
            interval(float): upper end of confidence interval.
        """
    
        alpha       = self.readFromSettings('confidenceInter')
        alpha       = 1 - alpha
        
        interval    = scipy.stats.norm.interval(alpha)[1]
        
        return interval
#        return 1.95996398454005404943 
    
        
    def _segmentT1(self):
        
#        magFrames   = self._t1MagnitudeFrames
#        
#        #If frames are wrapped
#        halfway     = int(len(magFrames)/2)
#        magFrames   = np.concatenate([magFrames[halfway:],
#                                      magFrames[:halfway]])
#        nclass = 3 #White, Grey, CSF
#        beta = 0.1 #between 0 and 0.5
#        
#        hmrf = tissue.TissueClassifierHMRF()
#        initSegment, finalSegment, PVE = hmrf.classify(magFrames,
#                                                       nclass,
#                                                       beta)
#        self._whiteMatterMask = PVE[:,:,2]
#        
#        self._mask = self._getMatchingSlice(self._whiteMatterMask)
        
        pass
        
    def _getMatchingSlice(self, arr):
        pass
    
        
    
    # ------------------------------------------------------------
    """Vessel Analysis"""
    
    def _getMedianDiameter(self):
        """Returns the diameter as specified in the settings."""
        
        diam = self.readFromSettings("medDiam")
        if diam is None:
            diam = 0
            
        diam = 53
        return diam
    
    
    def _calculateMedians(self):
        """Applies median filters to some necessary arrays.
        Starts a new process for each, to reduce processing time."""
        
        diameter = self._getMedianDiameter()
        
        velocityFrames  = np.asarray(self._selmaDicom.getVelocityFrames())
        magnitudeFrames = np.asarray(self._selmaDicom.getMagnitudeFrames())
        
        meanVelocityFrame       = np.mean(velocityFrames, axis=0)
        meanMagnitudeFrame      = np.mean(magnitudeFrames, axis=0)
        
        
        venc                = self._selmaDicom.getTags()['venc']
        phaseFrames         = velocityFrames * np.pi / venc
        complexSignal       = magnitudeFrames * (
                                                np.cos(phaseFrames) + 
                                                np.sin(phaseFrames) * 1j
                                                )
        realSignalSTD       = np.std(np.real(complexSignal), axis = 0, ddof=1)
        imagSignalSTD       = np.std(np.imag(complexSignal), axis = 0, ddof=1)
        
        rmsSTD              = np.sqrt( (realSignalSTD**2 + imagSignalSTD**2))
        
        
#        objList = [(diameter, meanVelocityFrame),
#                   (diameter, meanMagnitudeFrame),
#                   (diameter, rmsSTD)]
#        
#        nProcesses = min(cpu_count(), len(objList))
#        
#        freeze_support() #prevent multiprocessing from freezing
#        with Pool(nProcesses) as pool:
#            res = pool.map(applyMedianFilter, objList)
#            
#        self._medianVelocityFrame   = res[0]
#        self._medianMagnitudeFrame  = res[1]
#        self._medianRMSSTD          = res[2]
        
        
        self._medianVelocityFrame = scipy.signal.medfilt2d(meanVelocityFrame,
                                                           diameter)
        self._medianMagnitudeFrame= scipy.signal.medfilt2d(meanMagnitudeFrame,
                                                           diameter)
        self._medianRMSSTD          = scipy.signal.medfilt2d(rmsSTD,
                                                           diameter)
        
        
        
    def _subtractMedian(self):
        '''Find and subtract the median-filtered mean velocity frame from
        all velocity frames.'''
        
        velocityFrames                  = np.asarray(
                                        self._selmaDicom.getVelocityFrames())
        self._correctedVelocityFrames   = (velocityFrames -
                                        self._medianVelocityFrame)
    
    def _SNR(self):
        """Calculates the SNR in the velocity frames. This is done in the 
        following manner:
           
            First the velocity frames are converted to phase frames
            Next, the phase and magnitude frames are converted to a complex
                signal from which the standard deviation in the real and 
                imaginary component are calculated.
            Next, the root mean square of these standard deviations is obtained
                and a median-filter is applied.
            Next, the SNR in the magnitude frames is found.
            Lastly, the SNR in the velocity frames is calculated.            
        """
        
        magnitudeFrames     = np.asarray(
                                    self._selmaDicom.getMagnitudeFrames())
        magnitudeSNR        = div0(magnitudeFrames,
                                   self._medianRMSSTD)
        venc                = self._selmaDicom.getTags()['venc']
        
        
        self._velocitySTD   = venc / np.pi * div0(1, magnitudeSNR)
        self._velocitySNR   = np.mean(div0(self._correctedVelocityFrames,
                                                self._velocitySTD), axis=0)
        
        
    def _findSignificantFlow(self):
        """Uses the velocity SNR to find vessels with significant velocity."""
        sigma               = self.getSigma()
        self._sigFlowPos    = (self._velocitySNR > sigma).astype(np.uint8)
        self._sigFlowNeg    = (self._velocitySNR < -sigma).astype(np.uint8)
        self._sigFlow       = self._sigFlowNeg + self._sigFlowPos  
        
    def _removeZeroCrossings(self):
        """Removes all vessels where the flow changes sign."""
        
        velocityFrames  = np.asarray(
                            self._selmaDicom.getVelocityFrames())
        signs           = np.sign(velocityFrames)
        signdiff        = np.diff(signs, axis=0) 
        noZeroCrossings = np.sum(np.abs(signdiff), axis=0) == 0
        
        self._sigFlowPos *= noZeroCrossings
        self._sigFlowNeg *= noZeroCrossings
        self._sigFlow    *= noZeroCrossings
        
    def _findSignificantMagnitude(self):
        """
        Makes masks for all vessels with:
            -Positive magnitude
            -Negative magnitude
            -Isointense magnitude
        """          
        magnitudeFrames     = self._selmaDicom.getMagnitudeFrames()
        meanMagnitude       = np.mean(magnitudeFrames, axis = 0)
        sigma               = self.getSigma()
        
#        medianMagnitude     = scipy.signal.medfilt2d(meanMagnitude,
#                                                     self._medianDiameter)
        
        self._sigMagPos     = (meanMagnitude -
                               self._medianMagnitudeFrame -
                               sigma*self._medianRMSSTD
                                ) > 0
        self._sigMagPos     = self._sigMagPos.astype(np.uint8)
        
        self._sigMagNeg     = (meanMagnitude -
                               self._medianMagnitudeFrame +
                               sigma*self._medianRMSSTD
                                ) < 0
        self._sigMagNeg     = self._sigMagNeg.astype(np.uint8)
        
        self._sigMagIso     = self._sigFlow - self._sigMagNeg - self._sigMagPos
        self._sigMagIso     = (self._sigMagIso > 0).astype(np.uint8)
        
        
    def _applyT1Mask(self):
        """Applies the T1 mask (if any) to the sigFlowPos, sigFlowNeg and 
        sigFlow arrays."""
        mask = self._mask
        
        if mask is None:
            self._signalObject.errorMessageSignal.emit("No mask loaded.")
            return
        
        mask = mask.astype(bool) #prevent casting errors
        self._sigFlowPos *= mask
        self._sigFlowNeg *= mask
        self._sigFlow    *= mask
        
    def _removeGhosting(self):
        """
        Get xth percentile of vessels
        Cluster
        Go over each cluster
        if too small, ignore
        if normal: make small stripe
        if large: make large stripe
        
        remove stripes from mask
        
        
        """
        
        doGhosting      = self.readFromSettings('doGhosting')
        if not doGhosting:
            self._ghostingMask = np.zeros(self._mask.shape)
            
        
        #Read from settings
        percentile          = self.readFromSettings('brightVesselPerc')
        
        noVesselThresh      = self.readFromSettings('noVesselThresh')
        smallVesselThresh   = self.readFromSettings('smallVesselThresh')
        
        smallVesselExclX    = self.readFromSettings('smallVesselExclX')
        smallVesselExclY    = self.readFromSettings('smallVesselExclY')
        
        largeVesselExclX    = self.readFromSettings('largeVesselExclX')
        largeVesselExclY    = self.readFromSettings('largeVesselExclY')
        
        #Remove sharp edges from mean magnitude frame.
        magnitude       = self._selmaDicom.getMagnitudeFrames()
        meanMagnitude   = np.mean(magnitude, axis = 0)
        medianMagnitude = self._medianMagnitudeFrame
        meanMagnitude   -= medianMagnitude
        
        #Find threshold for 'bright' vessels and mask them.
        meanMagNonzero  = meanMagnitude(np.nonzero(meanMagnitude))
        threshold       = np.percentile(meanMagNonzero, percentile)
        brightVesselMask= (meanMagnitude > threshold)
        brightVesselMask= brightVesselMask.astype(np.uint8)
        
        #Cluster the bright vessels
        nClusters, clusters = cv2.connectedComponents(brightVesselMask)
        ghostingMask    = np.zeros(meanMagnitude.shape)
        
        #Go over all the clusters and add to ghostingMask
        for idx in range(nClusters):
            cluster = clusters == idx
            size    = np.sum(cluster)
            
            #If the cluster is too small, ignore
            if size <= noVesselThresh:
                continue
            
            elif size <= smallVesselThresh:
                #TODO:
                #find left, right, top and bottom
                #add buffer to left right,
                #Extend in y direction by [lenght[]]
                
                pass
            else:
                
                pass
                #same, but larger
            
            ghostingMask += cluster
            
        
        
        
        #store ghosting mask
        self._ghostingMask = ghostingMask
        
        
    def _removePerpendicularVessels(self):
        """
        Go over all clusters
        per cluster:
            Take window around it
            threshold magnitude on x percentage
            cluster result
            Find minor & major radius of main blob
            if major > x*minor, remove        
        
        """
        

    def _clusterVessels(self):
        """
        Uses the flow and magnitude classifications to cluser the vessels.
        The clusters are made per combination of velocity and magnitude 
        and are later added together.   
        
        TODO: Maybe change from open-cv to scipy:
            https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/
            scipy.ndimage.measurements.label.html
        
            That way, less packages in total need to be managed.
        """
        
        VNegMPos      = self._sigFlowNeg * self._sigMagPos
        VPosMPos      = self._sigFlowPos * self._sigMagPos
        VNegMNeg      = self._sigFlowNeg * self._sigMagNeg
        VPosMNeg      = self._sigFlowPos * self._sigMagNeg
        VNegMIso      = self._sigFlowNeg * self._sigMagIso
        VPosMIso      = self._sigFlowPos * self._sigMagIso   
        
        
        self._nComp   = 0
        self._labels  = np.zeros(self._sigFlow.shape,
                                 dtype = np.int32)
        
        #VNegMPos
        ncomp, labels = cv2.connectedComponents(VNegMPos.astype(np.uint8))
        self._nComp             += ncomp - 1
        labels[labels != 0]     += np.max(self._labels)
        self._labels            += labels
        
        #VPosMPos
        ncomp, labels = cv2.connectedComponents(VPosMPos.astype(np.uint8))
        self._nComp             += ncomp - 1
        labels[labels != 0]     += np.max(self._labels)
        self._labels            += labels
        
        #VNegMNeg
        ncomp, labels = cv2.connectedComponents(VNegMNeg.astype(np.uint8))
        self._nComp             += ncomp - 1
        labels[labels != 0]     += np.max(self._labels)
        self._labels            += labels
        
        #VPosMNeg
        ncomp, labels = cv2.connectedComponents(VPosMNeg.astype(np.uint8))
        self._nComp             += ncomp - 1
        labels[labels != 0]     += np.max(self._labels)
        self._labels            += labels
        
        #VNegMIso
        ncomp, labels = cv2.connectedComponents(VNegMIso.astype(np.uint8))
        self._nComp             += ncomp - 1
        labels[labels != 0]     += np.max(self._labels)
        self._labels            += labels
        
        #VPosMIso
        ncomp, labels = cv2.connectedComponents(VPosMIso.astype(np.uint8))
        self._nComp             += ncomp - 1
        labels[labels != 0]     += np.max(self._labels)
        self._labels            += labels
        
        
        #Write _labels to _vesselMask
        self._vesselMask        = (self._labels != 0)
        
    def _makeVesselDict(self):
        """Makes a dictionary containing the following statistics
        for each voxel in a vessel:
            -pixelID    (with arrays starting at 0)
            -row        (with arrays starting at 0)
            -column     (with arrays starting at 0)
            -clusternumber
            -VNeg       (true or false)
            -VPos       (true or false)
            -MPos       (true or false)
            -MIso       (true or false)
            -MNeg       (true or false)
            -Mean Magnitude
            -Magnitude STD
            -mean Velocity
            -Velocity STD
            -min Velocity
            -max Velocity
            -PI         (maxV - minV)/meanV
            -nPhases    (how many heart cycles)
            -Mag per cycle 
            -Velocity per cycle"""

        self._vesselDict = dict()        
        
        #Get some variables from memory to save time. 
        meanMagnitude   = np.mean(self._selmaDicom.getMagnitudeFrames(),
                                  axis = 0)
        meanVelocity    = np.mean(self._correctedVelocityFrames,
                                  axis = 0)
        magFrames       = np.asarray(self._selmaDicom.getMagnitudeFrames())
        
        
        #Keep track of the progress to emit to the progressbar
        i       = 0 
        total   = len(np.nonzero(self._labels)[0])
        
        for ncomp in range(1, self._nComp + 1):        #iterate over blobs
            for pixel in np.transpose(np.nonzero(self._labels == ncomp)):
                x,y = pixel
                value_dict = dict()
                value_dict['pixelID']       = int(y*self._labels.shape[-1] + x+1)
                value_dict['row']           = int(x+1)
                value_dict['column']        = int(y+1)
                value_dict['blob']          = int(self._labels[x, y])
                value_dict['vNeg']          = self._sigFlowNeg[x,y]
                value_dict['vPos']          = self._sigFlowPos[x,y]
                value_dict['mPos']          = self._sigMagPos[x,y] 
                value_dict['mIso']          = self._sigMagIso[x,y]
                value_dict['mNeg']          = self._sigMagNeg[x,y]
                value_dict['meanMag']       = meanMagnitude[x,y]
                value_dict['stdMagNoise']   = self._medianRMSSTD[x,y]
                value_dict['meanV']         = meanVelocity[x,y]
                value_dict['stdVNoise']     = np.mean(self._velocitySTD[:,x,y])
                value_dict['minV']          = np.min(self._correctedVelocityFrames[:,x,y])
                value_dict['maxV']          = np.max(self._correctedVelocityFrames[:,x,y])
                value_dict['PI']            = div0([(value_dict['maxV'] -  value_dict['minV'])],
                                              value_dict['meanV'])
                value_dict['nPhase']        = self._correctedVelocityFrames.shape[0]
                value_dict['magPerPhase']   = magFrames[:,x,y].tolist()
                value_dict['velPerPhase']   = self._correctedVelocityFrames[:,x,y].tolist()
                
                self._vesselDict[i] = value_dict
                
                #Emit progress to progressbar
                self._signalObject.setProgressBarSignal.emit(int(100 * i / total))
                i+= 1
        
        self._signalObject.setProgressBarSignal.emit(100)
        
    def _writeToFile(self):
        """
        Creates a filename for the output and passes it to writeVesselDict
        along with the vesselDict object to be written.
        """
        
        #Message if no vessels were found
        if np.nonzero(self._labels)[0] == 0:
            self._signalObject.errorMessageSignal.emit("No vessels Found")
            return
        
        #Get filename for textfile output
        fname = self._dcmFilename[:-4]
        fname += "-Vessel_Data.txt"
        
        SELMADataIO.writeVesselDict(self._vesselDict,
                                    fname)
        
        