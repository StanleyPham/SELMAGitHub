#!/usr/bin/env python

"""
This module is contains all the relevant classes that form the data layer 
of the SELMA project. It contains the following classes:

+ :class:`SELMADataObject`
    
"""

# ====================================================================
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.signal
import scipy.stats
import time
import cv2
#from multiprocessing import Pool, freeze_support, cpu_count

from PyQt5 import QtCore

# ====================================================================

import SELMADicom
import SELMAClassicDicom
import SELMAT1Dicom
import SELMADataIO
import SELMAGUISettings

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
    
    def __init__(self,
                 signalObject,
                 dcmFilename = None,
                 classic = False):
        
        self._mask          = None
        self._NBmask        = None      #Non binary mask, no trehshold applied
        self._t1            = None
        self._vesselMask    = None
        self._selmaDicom    = None
        
        if dcmFilename is not None:
            if classic:
                self._selmaDicom    = SELMAClassicDicom.SELMAClassicDicom(
                                                            dcmFilename)
                self._dcmFilename   = dcmFilename[0] + ".dcm"
            else:
                self._selmaDicom    = SELMADicom.SELMADicom(dcmFilename)
                self._dcmFilename   = dcmFilename 
            
        self._signalObject = signalObject
    
    '''Public'''
    
    # 
    # ------------------------------------------------------------------
    
    def analyseVessels(self):
        '''
        The main algorithm of segmenting and analysing the significant vessels.
        It is split in the following parts:
            -Preprocesses the data to be a gaussian around zero.
            -Find all significant voxels based on their SNR
            -Filter results
            -Cluster results into vessels
            -Extract and save vessel properties
        '''
        if self._selmaDicom is None:
            self._signalObject.errorMessageSignal.emit("No DICOM loaded.")
            return
        
        
        self._signalObject.setProgressBarSignal.emit(0)
        self._signalObject.setProgressLabelSignal.emit(
                    "Calculating median images")
        self._calculateMedians()
        self._signalObject.setProgressBarSignal.emit(60)
        self._signalObject.setProgressLabelSignal.emit(
                    "Finding significant vessels")
        self._subtractMedian()
                
        #Determine SNR of all voxels
        self._SNR()
        
        #Find all vessels with significant flow.
        self._findSignificantFlow()

        #Adjust and apply the Mask
        self._removeZeroCrossings()
        self._removeGhosting()
        self._removeOuterBand()
        self._updateMask()
        self._applyT1Mask()
        self._signalObject.setProgressBarSignal.emit(80)
        self._signalObject.setProgressLabelSignal.emit(
                    "Analysing clusters")
        
        #Cluster the vessels. 
        self._findSignificantMagnitude()
        self._clusterVessels()
        self._removeNonPerpendicular()
        self._deduplicateVessels()
        self._signalObject.setProgressBarSignal.emit(100)

#        Send vessels back to the GUI
        self._signalObject.sendVesselMaskSignal.emit(self._vesselMask)
        
        #make dictionary and write to disk
        self._signalObject.setProgressLabelSignal.emit(
                    "Writing results to disk")
        self._makeVesselDict()
        self._writeToFile()
    
        self._signalObject.setProgressLabelSignal.emit("")
        
        
        
    def segmentMask(self):
        if self._t1 is None:
            self._signalObject.errorMessageSignal.emit(
                    "Please load a t1 dicom first.")
            return
        
        self._signalObject.setProgressLabelSignal.emit(
                    "Segmenting white matter from T1 - This may take a while.")
        self._NBmask  = self._t1.getSegmentationMask()
        self._thresholdMask()
        self._signalObject.setProgressLabelSignal.emit(
                    "")
    
    #Getter functions
    # ------------------------------------------------------------------    
    def getFrames(self):
        return self._selmaDicom.getFrames()
    
    def getRawFrames(self):
        return self._selmaDicom.getRawFrames()
    
    def getNumFrames(self):
        return self._selmaDicom.getNumFrames()

    def getMask(self):
        if self._NBmask is None:
            return self._mask
        else:
            self._thresholdMask()
            return self._mask
    
    def getT1(self):
        return self._t1
    
    def getVenc(self):
        return self._selmaDicom.getTags()['venc']
    
    def getRescale(self):
        velFrames   = self._selmaDicom.getVelocityFrames()
        minres      = np.min(velFrames)
        maxres      = np.max(velFrames)
        
        return [minres, maxres]
    
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
        self._t1 = SELMAT1Dicom.SELMAT1Dicom(t1Fname, 
                                             self._selmaDicom.getDCM())
        
    def setVenc(self, venc):
        self._selmaDicom.setVenc(venc)
        
    def setVelRescale(self, rescale):
        self._selmaDicom.setVelRescale(rescale)
        
    '''Private'''
    # Setup data from .dcm file
    # ------------------------------------------------------------------    
    
    
    def _readFromSettings(self, key):
        """Loads the settings object associated with the program and 
        returns the value at the key."""
        
        COMPANY, APPNAME, _ = SELMAGUISettings.getInfo()
        COMPANY             = COMPANY.split()[0]
        APPNAME             = APPNAME.split()[0]
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
    
    def _getSigma(self):
        """ Returns the upper end of the confidence interval with the alpha
        value in the settings.
        
        Args:
            
        Returns:
            interval(float): upper end of confidence interval.
        """
    
        alpha       = self._readFromSettings('confidenceInter')
        alpha       = 1 - alpha
        
        interval    = scipy.stats.norm.interval(alpha)[1]
        
        return interval
    
        
    def _thresholdMask(self):
        #threshold the mask based on the value in the settings
        threshold   = self._readFromSettings("whiteMatterProb")
        self._mask  = np.copy(self._NBmask)
        self._mask[self._mask < threshold]  = 0
        self._mask[self._mask >= threshold] = 1
        self._mask = np.asarray(self._mask, dtype=int)
    
        
    
    # ------------------------------------------------------------
    """Vessel Analysis"""
    
    def _getMedianDiameter(self):
        """Returns the diameter as specified in the settings."""
        
        diam    = self._readFromSettings("medDiam")
        if diam is None:
            diam    = 0
            
        mmPix   = self._readFromSettings("mmPixel")
        if mmPix:
            ps      = self._selmaDicom.getPixelSpacing()
            newDiam = int(diam / ps)
            if newDiam % 2 == 0:
                newDiam += 1
            diam    = newDiam
            
        return diam
    
    
    def _calculateMedians(self):
        """Applies median filters to some necessary arrays.
        Starts a new process for each, to reduce processing time."""
        
        #Prepares the data to be filtered
        diameter = int(self._getMedianDiameter())
        
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
        
        #Multithreaded version, not very stable.
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
        
        
        #Either applies a gaussian smoothing filter or a median filter.
        #NOTE: the gaussian smoothing is not very reliable, should only
        #Be used for testing.
        gaussianSmoothing = self._readFromSettings('gaussianSmoothing')
        if gaussianSmoothing:
            #Find sigma from FWHM and median diameter in settings
            filterRadius    = int(diameter / 2.355)
            self._medianVelocityFrame   = gaussian_filter(meanVelocityFrame,
                                                          filterRadius)
            self._medianMagnitudeFrame  = gaussian_filter(meanMagnitudeFrame,
                                                          filterRadius)
            self._medianRMSSTD          = gaussian_filter(rmsSTD,
                                                          filterRadius)
        
        else:
            
            self._medianVelocityFrame   = scipy.signal.medfilt2d(
                                                        meanVelocityFrame,
                                                        diameter)
            self._medianMagnitudeFrame  = scipy.signal.medfilt2d(
                                                        meanMagnitudeFrame,
                                                        diameter)
            self._medianRMSSTD          = scipy.signal.medfilt2d(
                                                        rmsSTD,
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
        sigma               = self._getSigma()
        self._sigFlowPos    = (self._velocitySNR > sigma).astype(np.uint8)
        self._sigFlowNeg    = (self._velocitySNR < -sigma).astype(np.uint8)
        self._sigFlow       = self._sigFlowNeg + self._sigFlowPos  
        
        
    def _removeZeroCrossings(self):
        """Removes all vessels where the flow changes sign."""
        
        # velocityFrames  = np.asarray(
        #                     self._selmaDicom.getVelocityFrames())
        # signs           = np.sign(velocityFrames)
        
        signs           = np.sign(self._correctedVelocityFrames)
        signdiff        = np.diff(signs, axis=0) 
        noZeroCrossings = np.sum(np.abs(signdiff), axis=0) == 0
        
        self._sigFlowPos *= noZeroCrossings
        self._sigFlowNeg *= noZeroCrossings
        self._sigFlow    *= noZeroCrossings
        
        
    def _removeGhosting(self):
        """
        Creates a ghostingmask that can be used to subtract the areas 
        around bright vessels from the main mask.
        
        This mask is found as follows:
            
        Get xth percentile of vessels, read x from settings
        Cluster the bright vessels
        Go over each cluster and decide what size it is
            < noVesselThresh                        -> Ignore
            > noVesselTresh & < smallVesselThresh   -> Small exclusion zone
            > noVesselTresh & > smallVesselThresh   -> Large exclusion zone
        Create exclusion zone by finding the left, right , top , and bottom 
            most voxels and adding the exclusion buffers 
        Add exclusion zone to ghostingMask
                    
        
        """
        
        doGhosting      = self._readFromSettings('doGhosting')
        if not doGhosting:
            self._ghostingMask = np.zeros(self._mask.shape)
            return
        
        #Read from settings
        percentile          = self._readFromSettings('brightVesselPerc')
        
        noVesselThresh      = self._readFromSettings('noVesselThresh')
        smallVesselThresh   = self._readFromSettings('smallVesselThresh')
        
        smallVesselExclX    = self._readFromSettings('smallVesselExclX')
        smallVesselExclY    = self._readFromSettings('smallVesselExclY')
        
        largeVesselExclX    = self._readFromSettings('largeVesselExclX')
        largeVesselExclY    = self._readFromSettings('largeVesselExclY')
        
        #Remove sharp edges from mean magnitude frame.
        magnitude       = self._selmaDicom.getMagnitudeFrames()
        meanMagnitude   = np.mean(magnitude, axis = 0)
        medianMagnitude = self._medianMagnitudeFrame
        meanMagnitude   -= medianMagnitude
        
        #Find threshold for 'bright' vessels and mask them.
        meanMagNonzero  = np.abs(meanMagnitude[np.nonzero(meanMagnitude)])
        threshold       = np.percentile(meanMagNonzero, percentile*100)
        brightVesselMask= (np.abs(meanMagnitude) > threshold)
        brightVesselMask= brightVesselMask.astype(np.uint8)
        
        #Cluster the bright vessels
        nClusters, clusters = cv2.connectedComponents(brightVesselMask)
        ghostingMask    = np.zeros(meanMagnitude.shape)
        
        #Go over all the clusters and add to ghostingMask
        for idx in range(1, nClusters):     #skip 0, that's the background
            cluster = clusters == idx
            size    = np.sum(cluster)
            
            #If the cluster is too small, ignore
            if size <= noVesselThresh:
                continue
            
            #find left, right, top and bottom of cluster
            clusterCoords   = np.nonzero(cluster)
            left            = np.min(clusterCoords[1])
            right           = np.max(clusterCoords[1])
            top             = np.min(clusterCoords[0])
            bottom          = np.max(clusterCoords[0])
            
            #Small Vessel
            if size <= smallVesselThresh:
                
                #add buffer to left right, extend along y axis
                newLeft         = int(max(left      - smallVesselExclX,
                                      0))
                newRight        = int(min(right     + smallVesselExclX,
                                      meanMagnitude.shape[0] ))
                newTop          = int(max(top       - smallVesselExclY,
                                      0))
                newBottom       = int(min(bottom    + smallVesselExclY,
                                      meanMagnitude.shape[1]))
                
            #Large Vessel                
            else:
                #add buffer to left right, extend along y axis
                
                #Expand window with values from settings
                newLeft         = int(max(left      - largeVesselExclX,
                                      0))
                newRight        = int(min(right     + largeVesselExclX,
                                      meanMagnitude.shape[0] ))
                newTop          = int(max(top       - largeVesselExclY,
                                      0))
                newBottom       = int(min(bottom    + largeVesselExclY,
                                      meanMagnitude.shape[1]))
                
            #increase cluster size
            exclZone        = np.zeros(cluster.shape)
            exclZone[newTop : newBottom, newLeft : newRight] = 1
            
            #Update the ghostingMask
            ghostingMask += exclZone
        
        
        #store ghosting mask
        ghostingMask        = ghostingMask > 0
        self._ghostingMask  = ghostingMask.astype(np.uint8)
#        self._signalObject.sendVesselMaskSignal.emit(self._ghostingMask)
        
    def _removeOuterBand(self):
        """
        Creates an exclusion mask around the outer edges of the image with a 
        certain width.
        """
        
        ignoreOuterBand         = self._readFromSettings('ignoreOuterBand')
        self._outerBandMask     = np.zeros(self._mask.shape)
        if not ignoreOuterBand:
            return
        
        band                            = 80    #TODO, get from settings
        self._outerBandMask[:band, :]   = 1
        self._outerBandMask[:, :band]   = 1
        self._outerBandMask[-band:, :]  = 1
        self._outerBandMask[:, -band:]  = 1
        
    def _updateMask(self):
        """
        Removes the exclusion zones found in removeGhosting and 
        removeNonPerpendicular from the mask.
        
        Sends the updated mask to the GUI.
        """
        
        mask            = self._mask.astype(bool)
        ghost           = self._ghostingMask.astype(bool)
        outer           = self._outerBandMask.astype(bool)
        
        #Make mask without ghosting
        ghost           = ghost * mask
        maskMinGhost    = mask  ^ ghost
        
        #Make mask without outer edge
        outer           = outer * mask
        maskMinOuter    = mask  ^ outer
        
        #Combine all masks
        mask            = maskMinGhost & maskMinOuter
        
        self._mask = mask.astype(np.uint8)
        self._signalObject.sendMaskSignal.emit(self._mask)

    
    def _applyT1Mask(self):
        """First normalises, then applies the T1 mask (if any) to the 
        sigFlowPos, sigFlowNeg and sigFlow arrays."""
        mask = self._mask
        
        if mask is None:
            self._signalObject.errorMessageSignal.emit("No mask loaded.")
            return
        
        mask = mask.astype(bool) #prevent casting errors
        self._sigFlowPos *= mask
        self._sigFlowNeg *= mask
        self._sigFlow    *= mask
    
        
    def _findSignificantMagnitude(self):
        """
        Makes masks for all vessels with:
            -Positive magnitude
            -Negative magnitude
            -Isointense magnitude
        """          
        magnitudeFrames     = self._selmaDicom.getMagnitudeFrames()
        meanMagnitude       = np.mean(magnitudeFrames, axis = 0)
        sigma               = self._getSigma()
        
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
        
        
        self._nComp     = 0
        self._clusters  = []
        
        #VNegMPos
        ncomp, labels = cv2.connectedComponents(VNegMPos.astype(np.uint8))
        for comp in range(1,ncomp):
            self._clusters.append(labels == comp)
        
        #VPosMPos
        ncomp, labels = cv2.connectedComponents(VPosMPos.astype(np.uint8))
        for comp in range(1,ncomp):
            self._clusters.append(labels == comp)
            
        #VNegMNeg
        ncomp, labels = cv2.connectedComponents(VNegMNeg.astype(np.uint8))
        for comp in range(1,ncomp):
            self._clusters.append(labels == comp)
            
        #VPosMNeg
        ncomp, labels = cv2.connectedComponents(VPosMNeg.astype(np.uint8))
        for comp in range(1,ncomp):
            self._clusters.append(labels == comp)
            
        #VNegMIso
        ncomp, labels = cv2.connectedComponents(VNegMIso.astype(np.uint8))
        for comp in range(1,ncomp):
            self._clusters.append(labels == comp)
            
        #VPosMIso
        ncomp, labels = cv2.connectedComponents(VPosMIso.astype(np.uint8))
        for comp in range(1,ncomp):
            self._clusters.append(labels == comp)        
        
        #Write _clusters to _vesselMask
        self._createVesselMask()
        
        
        #Cluster only significant magnitude
        _, self._posMagClusters     = cv2.connectedComponents(
                                        self._sigMagPos)
        self._posMagClusters       *= self._mask
        _, self._negMagClusters     = cv2.connectedComponents(
                                        self._sigMagNeg)
        self._negMagClusters       *= self._mask
        
        

    def _createVesselMask(self):
        """
        Iterates over the clusters found in _clusters and creates
        a mask of all the vessels.
        """
        
        mask = np.zeros(self._mask.shape,
                        dtype = np.int32)
        
        for labels in self._clusters:
            mask += labels
        
        self._vesselMask        = mask.astype(bool)
        
    
    
    def _removePerpendicularTine(self):
        
        """
        According to tine:
            -Data to work on:
                tempim = mean(magdata, 3); %image to be drawn on
                tempmean = mean(tempim(:));
                tempstd = std(tempim(:));
                tempmin = tempmean-1*tempstd;
                tempmax = tempmean+3*tempstd;
                tempscaled = (tempim-tempmin)./(tempmax-tempmin); 
                tempscaled(tempscaled<0)=0; 
                tempscaled(tempscaled>1)=1;
                
            -only look at Mpos
            -take 15x15 window centered around blob centre
            -threshold 0.8*blobcentre value
            -cluster the results
            -Find the blob closest to the centre
            -find the major & minor axis lengths
        
        """
        meanMagnitude   = np.mean(self._selmaDicom.getMagnitudeFrames(),
                                  axis = 0)
        stdMagnitude    = np.std(self._selmaDicom.getMagnitudeFrames(),
                                  axis = 0)
        minMagnitude    = meanMagnitude - stdMagnitude
        maxMagnitude    = meanMagnitude + 3*stdMagnitude
        scaledMagnitude = ((meanMagnitude - minMagnitude) / 
                           (maxMagnitude - minMagnitude))
        
        
        for cluster in self._posMagClusters:
            pass
        
        
    
    def _removeNonPerpendicular(self):
        """
        Go over all clusters
        per cluster:
            Take window around it
            threshold magnitude on x percentage
            cluster result
            Find minor & major radius of main blob
            if major > x*minor, remove        
        
        """
        
        removeNonPerp      = self._readFromSettings('removeNonPerp')
        if not removeNonPerp:
            return
        
        #Get other values from settings
        removePerpX             = self._readFromSettings('removePerpX')
        removePerpY             = self._readFromSettings('removePerpY')
        removePerpMagThresh     = self._readFromSettings('removePerpMagThresh')
        removePerpRatioThresh   = self._readFromSettings('removePerpRatioThresh')
        
        #Get mean magnitude frame
        magnitudeFrames         = self._selmaDicom.getMagnitudeFrames()
        meanMagnitude           = np.mean(magnitudeFrames, axis = 0)
        
        #Iterate over clusters:
        for idx, cluster in enumerate(self._clusters):
            
            #find left, right, top and bottom of cluster
            clusterCoords   = np.nonzero(cluster)
            left            = np.min(clusterCoords[1])
            right           = np.max(clusterCoords[1])
            top             = np.min(clusterCoords[0])
            bottom          = np.max(clusterCoords[0])
            
            #Expand window with values from settings
            left            = int(max(left - removePerpX, 0))
            right           = int(min(right + removePerpX,
                                      meanMagnitude.shape[0]))
            top             = int(max(top - removePerpY, 0))
            bottom          = int(min(bottom + removePerpY,
                                      meanMagnitude.shape[1]))
            
            #Get magnitude voxels in window around cluster:
            magWindow       = meanMagnitude[top  : bottom,
                                            left : right]
            
            #threshold
            #TODO, compare to matlab
            threshold       = np.percentile(magWindow, removePerpMagThresh * 100)
            magWindowThresh = magWindow > threshold
            
            #cluster the result
            ncomp, labels   = cv2.connectedComponents(
                                        magWindowThresh.astype(np.uint8))
            
            #find largest (nonzero) cluster
            counts          = np.bincount(labels[np.nonzero(labels)])
            largestCluster  = np.argmax(counts)
            
            #find major and minor radius of cluster
            blob        = labels == largestCluster
            contours,_  = cv2.findContours(blob.astype(np.uint8), 1, 1)
            cnt         = contours[0]
            try:
                ellipse     = cv2.fitEllipse(cnt)
                rad1, rad2  = ellipse[1]
                majorRad    = max(rad1, rad2)
                minorRad    = min(rad1, rad2)
            except:
                #if fitEllipse crashes because the contour size is too small,
                #assume that it's a round vessel
                continue
            
            
            #Remove cluster from list if ellipse not circular enough
            if minorRad == 0:
                del(self._clusters[idx])
                continue
            
            if majorRad / minorRad > removePerpRatioThresh:
                del(self._clusters[idx])
                
        #Add the edited clusters to self._vesselMask
        self._createVesselMask()
        
    def _deduplicateVessels(self):
        """
        Tine's version:
            
            Take the first voxel of each cluster
            check whether any of them are <6 pixels apart
            if so, remove both clusters
        
        """
        pass

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
            -iMblob     (magnitude clustering list)
            -Mag per cycle 
            -Velocity per cycle"""

        self._vesselDict = dict()        
        
        #Get some variables from memory to save time. 
        meanMagnitude   = np.mean(self._selmaDicom.getMagnitudeFrames(),
                                  axis = 0)
        meanVelocity    = np.mean(self._correctedVelocityFrames,
                                  axis = 0)
        magFrames       = np.asarray(self._selmaDicom.getMagnitudeFrames())
        
        #TODO: check if needs to be multiplication.
        iMblob          = self._posMagClusters - self._negMagClusters 
        
        
        #Keep track of the progress to emit to the progressbar
        i       = 0 
        total   = np.sum(np.asarray(self._clusters))
        
        for idx, cluster in enumerate(self._clusters):
            
            
            #Sort pixels in cluster by mean velocity (largest to smallest)
            pixels  = np.nonzero(cluster)
            velocities  = np.abs(meanVelocity[pixels])
            indexes     = np.argsort(velocities)
            indexes     = indexes[::-1]    #largest to smallest            
            
            pixels      = np.transpose(pixels)
            
            for num, pidx in enumerate(indexes):
                x,y = pixels[pidx]
                value_dict = dict()
                value_dict['pixel']         = int(y*cluster.shape[-1] + x+1)
                value_dict['ir']            = int(x+1)
                value_dict['ic']            = int(y+1)
                value_dict['iblob']         = int(idx + 1)
                value_dict['ipixel']        = int(num + 1)
                value_dict['Vneg']          = round(self._sigFlowNeg[x,y],  4)
                value_dict['Vpos']          = round(self._sigFlowPos[x,y],  4)
                value_dict['Mpos']          = round(self._sigMagPos[x,y],   4)
                value_dict['Miso']          = round(self._sigMagIso[x,y],   4)
                value_dict['Mneg']          = round(self._sigMagNeg[x,y],   4)
                value_dict['meanMag']       = round(meanMagnitude[x,y],     4)
                value_dict['stdMagnoise']   = round(self._medianRMSSTD[x,y],4)
                value_dict['meanV']         = round(meanVelocity[x,y],      4)
                value_dict['stdVnoise']     = round(np.mean(
                                                self._velocitySTD[:,x,y]),  4)
                value_dict['minV']          = round(np.min(np.abs(
                                        self._correctedVelocityFrames[:,x,y])),
                                                    4)
                value_dict['maxV']          = round(np.max(np.abs(
                                        self._correctedVelocityFrames[:,x,y])),
                                                    4)
                value_dict['PI']            = abs(round(div0(
                                             [(value_dict['maxV'] -
                                              value_dict['minV'])],
                                              value_dict['meanV'])[0],
                                                    4))
                value_dict['nPha']          = self._correctedVelocityFrames.shape[0]
                value_dict['imBlob']        = int(iMblob[x,y])
                #Magnitude per phase
                for num, value in enumerate(magFrames[:,x,y].tolist()):
                    num += 1
                    if num < 10:
                        numStr = '0' + str(num)
                    else:
                        numStr = str(num)
                        
                    value_dict['Mpha' + numStr] = round(value, 4)
                
                #Velocity per phase
                for num, value in enumerate(
                        self._correctedVelocityFrames[:,x,y].tolist()):
                    num += 1
                    if num < 10:
                        numStr = '0' + str(num)
                    else:
                        numStr = str(num)
                        
                    value_dict['Vpha' + numStr] = round(value, 4)
                
                #Add vesselinfo to vessel dictionary
                self._vesselDict[i] = value_dict
                
                #Emit progress to progressbar
                self._signalObject.setProgressBarSignal.emit(
                        int(100 * i / total))
                i+= 1
        
        self._signalObject.setProgressBarSignal.emit(100)
        
    def _writeToFile(self):
        """
        Creates a filename for the output and passes it to writeVesselDict
        along with the vesselDict object to be written.
        """
        
        #Message if no vessels were found
        if len(np.nonzero(self._clusters)[0]) == 0:
            self._signalObject.errorMessageSignal.emit("No vessels Found")
            return
        
        #Get filename for textfile output
        fname = self._dcmFilename[:-4]
        fname += "-Vessel_Data.txt"
        
        addonDict = self.getAddonDict()
        
        SELMADataIO.writeVesselDict(self._vesselDict,
                                    addonDict,
                                    fname)
        
    def getAddonDict(self):
        """Makes a dictionary that contains the necessary information for
        repeating the analysis.""" 
        
        COMPANY, APPNAME, version = SELMAGUISettings.getInfo()
        COMPANY             = COMPANY.split()[0]
        APPNAME             = APPNAME.split()[0]
        version             = version.split()[0]
        settings            = QtCore.QSettings(COMPANY, APPNAME)
        
        addonDict   = dict()
        
        for key in settings.allKeys():
            addonDict[key]  = settings.value(key)
        
        venc                = self._selmaDicom.getTags()['venc']
        addonDict['venc']   = venc
        addonDict['version']= version
        
        date                = time.localtime()
        datestr     = str(date[2]) + '/' + str(date[1]) + '/' + str(date[0])
        timestr     = str(date[3]) + ':' + str(date[4]) + ':' + str(date[5])
        addonDict['date']   = datestr
        addonDict['time']   = timestr
        
        addonDict['filename'] = self._dcmFilename
        
        return addonDict
        
        
        
        