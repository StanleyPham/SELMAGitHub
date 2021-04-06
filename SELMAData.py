#!/usr/bin/env python

"""
This module is contains all the relevant classes that form the data layer 
of the SELMA project. It contains the following classes:

+ :class:`SELMADataObject`
    
"""

# ====================================================================
import numpy as np
from skimage import measure 
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

def remove_ccs_from_mask(entry_mask,conditional_mask):
    """This function is based on the version found in the original MATLAB
    implementation (line 1745 in PulsateGUI). This function finds the overlap
    between the flow mask and the significant magnitude mask. Coordinates of
    overlapping clusters are identified and the entire overlapping cluster is
    removed from the flow mask. The remaining flow mask is returned for the
    next magnitude direction."""
    
    def intersection(arrA, arrB):
        """ Checks for intersection between two 2D arrays"""
        #from: https://stackoverflow.com/questions/24477270/python-intersection-of-2d-numpy-arrays
    
        return not set(map(tuple, arrA)).isdisjoint(map(tuple, arrB))

    entry_ncomp, entry_mask_labels = cv2.connectedComponents(entry_mask) #identify clusters in flow mask
    entry_mask_stats = measure.regionprops_table(entry_mask_labels,properties=('label','coords'))
    entry_blob_coords = entry_mask_stats['coords'] #extract coordinates of all pixels belonging in clusters
    
    #skip if there is no overlap between flow mask and significant magnitude mask
    if not any(map(len,np.nonzero(entry_mask*conditional_mask))):
        
        output_mask = entry_mask #output entry mask directly
        
        return output_mask
    
    # extract pixel coordinates of overlapping clusters
    overlap_mask_stats = measure.regionprops_table((entry_mask*conditional_mask).astype(np.uint8),properties=('label','coords'))
    overlap_blob_coords = overlap_mask_stats['coords'][0]
    
    # Might be useful in the future to document which clusters are overlapping
    identified_blobs = []
    
    for blob in range(0,entry_ncomp - 1): # iterate over clusters in flow mask
    
        # if any of the coordinates of the overlapping clusters correspond 
        # with the cluster coordinates of the flow mask, the entire cluster is
        # removed from the flow mask
        if intersection(entry_blob_coords[blob],overlap_blob_coords) == 1:
        
            entry_blob_coords[blob] = [] #remove cluster from flow mask
        
            # keep track of overlapping clusters
            identified_blobs = np.append(identified_blobs,blob).astype(int)
    
    output_mask = np.zeros(entry_mask.shape)
    
    # Create new output mask which only contains the clusters with significant
    # flow but no longer significant magnitude in the given direction (pos or 
    # neg). The new output mask will be used for the next magnitude direction
    for blob in range(0,len(entry_blob_coords)):
    
        if entry_blob_coords[blob]  != []:
        
            output_mask[entry_blob_coords[blob][:,0],entry_blob_coords[blob][:,1]] = 1
            
    return output_mask

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
        The main algorithm of segmenting & analysing the significant vessels.
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
        self._calculatePI()
        self._createVesselMask()
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
        return self._vesselDict, self._velocityDict
    
    def getBatchAnalysisResults(self):
        
        self._makeBatchAnalysisDict()
        
        return self._batchAnalysisDict
    
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
            self._signalObject.errorMessageSignal.emit(
                "Wrong setting accessed.")
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
  
        # if int(self._selmaDicom._DCM.MagneticFieldStrength) == 3:
            
        #     alpha = 0.99
        #     interval    = scipy.stats.norm.interval(alpha)[1]
            
        #     return interval
               
        alpha       = self._readFromSettings('confidenceInter') #0.05
        alpha       = 1 - alpha
      
        RR_interval = self._selmaDicom.getRRIntervals()
        
        interval    = scipy.stats.norm.interval(alpha)[1]
        
        # interval = 2
        
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
        
        velocityFrames  = np.asarray(self._selmaDicom.getVelocityFrames()) #phase Frames are used in the 3T Test Retest data
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
            Next, the root mean square of these standard deviations is 
            obtained and a median-filter is applied.
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
        
        # np.save(self._dcmFilename[0:61] + '_velocitySNR.npy',self._velocitySNR)
        
        # import pdb; pdb.set_trace() #to catch and save VNR data for histograms
        
    def _findSignificantFlow(self):
        """Uses the velocity SNR to find vessels with significant velocity."""
        
        sigma               = self._getSigma()
        
        #PULSATE_meanstd = 0.5184721506309825 or 0.49234149129113325 (mirrored)
        
        # PULSATE_meanstd = 0.49234149129113325
        
        # VNR_BG = self._velocitySNR * self._mask
    
        # self._mask[abs(VNR_BG) > 3] = 0
    
        # voxel_coordinates = np.where(self._mask == 1)
    
        # voxels_in_mask = np.zeros((1,len(voxel_coordinates[0])))
 
        # for j in range(0,len(voxel_coordinates[0])):
        
        #     voxels_in_mask[0,j] = self._velocitySNR[voxel_coordinates[0][j],voxel_coordinates[1][j]]
            
        # voxels_in_mask = np.sort(voxels_in_mask)
        
        # voxels = np.zeros((1,2 * min(enumerate(voxels_in_mask[0,:]), key = lambda x: abs(x[1] - 0))[0]))    
    
        # voxels[0,0:min(enumerate(voxels_in_mask[0,:]), key = lambda x: abs(x[1] - 0))[0]] = voxels_in_mask[0,0:min(enumerate(voxels_in_mask[0,:]), key = lambda x: abs(x[1] - 0))[0]]
        # dummy = abs(voxels_in_mask[0,0:min(enumerate(voxels_in_mask[0,:]), key = lambda x: abs(x[1] - 0))[0]])
        # voxels[0,min(enumerate(voxels_in_mask[0,:]), key = lambda x: abs(x[1] - 0))[0]:len(voxels[0,:])] = np.fliplr(dummy.reshape((1,len(dummy))))
        
        # voxels_in_mask = voxels
    
        # (mu_norm, sigma_norm) = scipy.stats.norm.fit(voxels_in_mask)
        
        # sigma = sigma * (sigma_norm / PULSATE_meanstd)
        
        # self._mask[abs(VNR_BG) > 3] = 1
        
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
        
        #self._sigMagIso = self._sigFlow - self._sigMagNeg - self._sigMagPos
        
        # Consistent with MATLAB
        self._sigMagIso = (self._sigMagPos == 0) * (self._sigMagNeg == 0)
        self._sigMagIso = (self._sigMagIso > 0).astype(np.uint8)
        
    def _clusterVessels(self):
        """
        Uses the flow and magnitude classifications to cluser the vessels.
        The clusters are made per combination of velocity and magnitude 
        and are later added together.   
        
        TODO: Maybe change from open-cv to scipy:
            https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/
            scipy.ndimage.measurements.label.html
        
            That way, less packages in total need to be managed.
            
        01-02-2021: Turned off clustering of significant flow with isointense
                    magnitude since this causes erroneous detection of extra
                    vessels. A future use case for these clusters should be 
                    discussed, otherwise these lines can be removed.
        18-02-2021: Changed the clustering algorithm to better match the 
                    original MATLAB implementation. Flow masks for significant
                    flow in both directions are created. Clusters with a
                    significant magnitude in either direction are appended to
                    the detected clusters and are stepwise removed from flow
                    masks. The algorithm first checks positive magnitude
                    clusters and then negative magnitude clusters and removes
                    them from flow mask if the magnitude is significant. The
                    remaining clusters are then grouped as isointense.
        """
        
        ## DEPRECIATED: might become useful if we want to build in cluster selection
        # PositiveMagnitude            = self._readFromSettings('PositiveMagnitude')
        # NegativeMagnitude            = self._readFromSettings('NegativeMagnitude')
        # IsointenseMagnitude          = self._readFromSettings('IsointenseMagnitude')
        
        self._nComp     = 0
        self._clusters  = []
        
        'Positive magnitude clustering'
        
        # original flow masks with significant magnitudes in either direction
        entry_mask_VNeg = self._sigFlowNeg 
        entry_mask_VPos = self._sigFlowPos
    
        # Mask containing positive or negative flow with positive magnitude
        VNegMPos      = entry_mask_VNeg.astype(np.uint8) * self._sigMagPos
        VPosMPos      = entry_mask_VPos.astype(np.uint8) * self._sigMagPos
        
        #VNegMPos
        # Find clusters with negative flow and postive magnitude
        ncomp_VNegMPos, labels = cv2.connectedComponents(VNegMPos.astype(np.uint8))
        
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VNegMPos):
            self._clusters.append(labels == comp)
        
        # Remove found and overlapping clusters from flow mask
        output_mask_VNeg = remove_ccs_from_mask(entry_mask_VNeg,VNegMPos)
        
        #VPosMPos
        # Find clusters with positive flow and postive magnitude
        ncomp_VPosMPos, labels = cv2.connectedComponents(VPosMPos.astype(np.uint8))
        
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VPosMPos):
            self._clusters.append(labels == comp)
        
        # Remove found and overlapping clusters from flow mask
        output_mask_VPos = remove_ccs_from_mask(entry_mask_VPos,VPosMPos)
            
        self._NoMPosClusters = (ncomp_VNegMPos - 1) + (ncomp_VPosMPos - 1)
        
        'Negative magnitude clustering'
        
        # Output masks of positive magnitude clustering is input for the
        # negative magnitude clustering. Flow masks with only negative 
        # significant magnitudes are remaining    
        entry_mask_VNeg = output_mask_VNeg.astype(np.uint8)
        entry_mask_VPos = output_mask_VPos.astype(np.uint8)
        
        # Mask containing positive or negative flow with negative magnitude
        VNegMNeg      = entry_mask_VNeg.astype(np.uint8) * self._sigMagNeg
        VPosMNeg      = entry_mask_VPos.astype(np.uint8) * self._sigMagNeg
        
        #VNegMNeg
        # Find clusters with negative flow and negative magnitude
        ncomp_VNegMNeg, labels = cv2.connectedComponents(VNegMNeg.astype(np.uint8))
        
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VNegMNeg):
            self._clusters.append(labels == comp)
        
        # Remove found and overlapping clusters from flow mask
        output_mask_VNeg = remove_ccs_from_mask(entry_mask_VNeg,VNegMNeg)
        
        #VPosMNeg
        # Find clusters with positive flow and negative magnitude
        ncomp_VPosMNeg, labels = cv2.connectedComponents(VPosMNeg.astype(np.uint8))
        
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VPosMNeg):
            self._clusters.append(labels == comp)
         
        # Remove found and overlapping clusters from flow mask
        output_mask_VPos = remove_ccs_from_mask(entry_mask_VPos,VPosMNeg)
        
        self._NoMNegClusters = (ncomp_VNegMNeg - 1) + (ncomp_VPosMNeg - 1)
        
        'Isointense magnitude clustering'

        # Output masks of negative magnitude clustering is input for the
        # isointense magnitude clustering. Flow masks with neither positive or 
        # negative significant magnitude are remaining.
        entry_mask_VNeg = output_mask_VNeg.astype(np.uint8)
        entry_mask_VPos = output_mask_VPos.astype(np.uint8)
        
        # Mask containing positive or negative flow with isointense magnitude
        VNegMIso      = entry_mask_VNeg.astype(np.uint8) * self._sigMagIso
        VPosMIso      = entry_mask_VPos.astype(np.uint8) * self._sigMagIso
        
        #VNegMIso
        # Find clusters with negative flow and iosintense magnitude
        ncomp_VNegMIso, labels = cv2.connectedComponents(VNegMIso.astype(np.uint8))
        
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VNegMIso):
            self._clusters.append(labels == comp)
            
        # Remove found and overlapping clusters from flow mask    
        output_mask_VNeg = remove_ccs_from_mask(entry_mask_VNeg,VNegMIso)
        
        #VPosMIso
        # Find clusters with positive flow and iosintense magnitude
        ncomp_VPosMIso, labels = cv2.connectedComponents(VPosMIso.astype(np.uint8))
        
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VPosMIso):
            self._clusters.append(labels == comp)
        
        # Remove found and overlapping clusters from flow mask                            
        output_mask_VPos = remove_ccs_from_mask(entry_mask_VPos,VPosMIso)
        
        self._NoMIsoClusters = (ncomp_VNegMIso - 1) + (ncomp_VPosMIso - 1)

        #Cluster only significant magnitude (remainder from v1.0)
        _, self._posMagClusters     = cv2.connectedComponents(
                                        self._sigMagPos * self._mask)
        _, self._negMagClusters     = cv2.connectedComponents(
                                        self._sigMagNeg * self._mask)
    
    def _removeNonPerpendicular(self):
        
        """
        Finds the non-perpendicular vessels and removes them. Algorithm works
        as follows:
            -Rescale magnitude image
            -Iterate over clusters:
                -if only posivitve mag. clusters: select for those
                -Find centre of cluster
                -Select window around cluster
                -Threshold based on centre intensity
                -Find connected components in thresholded window
                -Take the one closest to the centre
                -Find contour of component
                -Fit ellipse
                -Determine ratio major/minor axis
                -Remove cluster based on ratio
        
        """
        if not self._readFromSettings('removeNonPerp'):
            # Added clauses for seperate scenarios when different settings are
            # turned on or off. This ensures the correct clusters are passed
            # through to the end
            
            self._perp_clusters = []
            self._non_perp_clusters = []
    
            return 
        
        self._non_perp_clusters = []
        self._perp_clusters = []
        self._axes_ratio = []
        
        onlyMPos            = self._readFromSettings('onlyMPos')
        minScaling          = self._readFromSettings('minScaling')
        maxScaling          = self._readFromSettings('maxScaling')
        winRad              = int(self._readFromSettings('windowSize'))
        magnitudeThresh     = self._readFromSettings('magnitudeThresh')
        ratioThresh         = self._readFromSettings('ratioThresh')
        
        meanMagnitude   = np.mean(self._selmaDicom.getMagnitudeFrames(),
                                  axis = 0)
        stdMagnitude    = np.std(self._selmaDicom.getMagnitudeFrames())
        stdMagnitude_MATLAB    = np.std(meanMagnitude)
        
        # MATLAB determines the std using the mean magnitude frame averaged
        # over the entire cardiac cycle (spatial variance). SELMA determines 
        # the std using all magnitude frames of the entire cardiac cycle
        # (temporal variance). The difference in std is about 1-2% between the
        # two implementations and has almost no effect on the amount of 
        # detected vessels. 
        
        meanVelocity    = np.mean(self._correctedVelocityFrames,
                                  axis = 0)
        
        #Rescale magnitude image
        meanMeanMag     = np.mean(meanMagnitude)
        minMagnitude    = meanMeanMag - minScaling * stdMagnitude
        maxMagnitude    = meanMeanMag + maxScaling * stdMagnitude
        scaledMagnitude = ((meanMagnitude - minMagnitude) / 
                           (maxMagnitude - minMagnitude))   
        scaledMagnitude[scaledMagnitude > 1] = 1
        scaledMagnitude[scaledMagnitude < 0] = 0
   
        for idx, cluster in enumerate(self._clusters):
            
            if onlyMPos:
                #Find the voxel with the highest velocity and check whether
                #it is Mpos, if not continue to the next cluster
                pixels      = np.nonzero(cluster)
                velocities  = np.abs(meanVelocity[pixels])
                indexes     = np.argsort(velocities)
                x,y         = np.transpose(pixels)[indexes[-1]]
                
                if not self._sigMagPos[x,y]:
                    
                    continue
           
            if np.size(np.where(cluster)[0]) > 1: 
                # Check if cluster is larger than 1 voxel. If not, assume it 
                # is a round vessel
            
                #find centre coordinate of cluster (row column)
                clusterCoords   = np.nonzero(cluster)
                centre          = [int(np.mean(clusterCoords[0]) + 0.5),
                                   int(np.mean(clusterCoords[1]) + 0.5)] 
                
                # int() always rounds down regardless of decimal value. This 
                # creates unintended behaviour where centre coordinates could
                # be off by 1 pixel. This is fixed by adding 0.5 to ensure
                # rounding is always correct
                
                #Get window around cluster in magnitude image
                magWindow       = scaledMagnitude[centre[0] - winRad:
                                                  centre[0] + winRad,
                                                  centre[1] - winRad:
                                                  centre[1] + winRad ]
                    
                #Threshold window to gain magnitude clusters of bright voxels
                threshold       = scaledMagnitude[centre[0], centre[1]]
                threshold       *= magnitudeThresh         
                magWindowThresh = (magWindow >= threshold).astype(np.uint8)
                
                #Find cluster closest to centre
                ncomp, labels   = cv2.connectedComponents(magWindowThresh)
                distances   = []
                for n in range(1, ncomp):
                    distances.append(
                        np.sqrt(
                            (np.mean(np.nonzero(labels == n)[0]) - winRad)**2 +
                            (np.mean(np.nonzero(labels == n)[1]) - winRad)**2))
                blob = labels == np.argmin(distances) + 1
                
                # New attempt at determining blob shape using regionprops. Now
                # more in line with MATLAB implementation. However, it is not 
                # exactly the same. Edge cases might exist where the axes ratio 
                # in MATLAB is < 2 but in SELMA it is > 2.
                    
                blob_stats = measure.regionprops_table(blob.astype(np.uint8),properties=('centroid','minor_axis_length','major_axis_length'))
                
                minorRad = blob_stats['minor_axis_length'][0]
                majorRad = blob_stats['major_axis_length'][0]
                
                self._axes_ratio.append(majorRad/minorRad)
                  
                'OLD METHOD'
                # #Find contour of blob, if multiple, concatenate them
                # contours,_  = cv2.findContours(blob.astype(np.uint8), 1, 1)
                # cnt         = np.concatenate(contours)
                # try:
                #     ellipse     = cv2.fitEllipse(cnt)
                #     rad1, rad2  = ellipse[1]
                #     majorRad    = max(rad1, rad2)
                #     minorRad    = min(rad1, rad2)
                    
                # except:
                #     #if fitEllipse crashes because the contour size is too small,
                #     #assume that it's a round vessel
                    
                #     #self._perp_clusters.append(self._clusters[idx])
                    
                #     continue
                
                #Remove cluster from list if ellipse not circular enough
                # if minorRad == 0:
                    
                    # self._non_perp_clusters.append(self._clusters[idx])
                    
                    # #del(self._clusters[idx])
                    
                    # continue
                
                if majorRad / minorRad > ratioThresh:
                    
                    self._non_perp_clusters.append(self._clusters[idx])
                    
                else:
                    
                    self._perp_clusters.append(self._clusters[idx])
                    
            else:
                
                self._axes_ratio.append(1)
                
                self._perp_clusters.append(self._clusters[idx])

        self._Noperp_clusters = len(self._perp_clusters)
                                         
    def _deduplicateVessels(self):
        """         
            Take the first voxel of each cluster
            check whether any of them are <6 pixels apart
            if so, remove both clusters
        """
        
        # Added clauses for seperate scenarios when different settings are
        # turned on or off. This ensures the correct clusters are passed
        # through to the end

        if not self._readFromSettings('removeNonPerp'):
            
            clusters = self._clusters
            
        if self._readFromSettings('removeNonPerp'):
            
            clusters = self._perp_clusters

        if not self._readFromSettings('deduplicate') and not self._readFromSettings('removeNonPerp'):
            
            self._lone_vessels = self._clusters
            self._cluster_vessels = []
            
            return
        
        if not self._readFromSettings('deduplicate') and self._readFromSettings('removeNonPerp'):
            
            self._lone_vessels = self._perp_clusters
            self._cluster_vessels = []
            
            return
        
        self._lone_vessels = clusters
        self._cluster_vessels = []
        
        dedupRange  = self._readFromSettings('deduplicateRange')
        
        #First make a list of all the voxels with the highest velocity per
        #cluster
        meanVelocity    = np.mean(self._correctedVelocityFrames,
                                  axis = 0)
        voxels  = []
        
        for cluster in self._lone_vessels:
           
            #Find the voxel with the highest velocity and check whether
            #it is Mpos, if not continue to the next cluster
            pixels      = np.nonzero(cluster)
            velocities  = np.abs(meanVelocity[pixels])
            indexes     = np.argsort(velocities)
            x,y         = np.transpose(pixels)[indexes[-1]]
        
            voxels.append([x,y])
            
        voxels  = np.asarray(voxels)

        if not voxels.size:
            return
        
        #Next, create matrix of the distances between all these voxels
        x       = np.repeat(np.reshape(voxels[:,0],(-1,1)), len(voxels), 1)
        xArr    = (x - np.transpose(x))**2
        
        y       = np.repeat(np.reshape(voxels[:,1],(-1,1)), len(voxels), 1)
        yArr    = (y - np.transpose(y))**2
        
        distances   = np.sqrt(xArr + yArr)
        selection   = np.tril((distances != 0) * (distances < dedupRange))
        idx         = np.unique(np.nonzero(selection))
   
        #Remove the selected clusters
        for i, clusterNum in enumerate(idx):
            
            self._cluster_vessels.append(self._lone_vessels[clusterNum - i])
            
            del(self._lone_vessels[clusterNum - i])
            
    def _calculatePI(self):
        """
        Computes the average velocity over all the detected vessels and 
        computes the PI over all the vessels using the average normalised
        velocity. This implementation completely corresponds with the method
        found in MATLAB."""
        
        meanVelocity    = np.mean(self._correctedVelocityFrames,axis = 0)
        
        V_cardiac_cycle = np.zeros((len(self._lone_vessels),self._correctedVelocityFrames.shape[0] + 3))

        for idx, vessel in enumerate(self._lone_vessels):
        
            vesselCoords   = np.nonzero(vessel)

            vessel_velocities = abs(meanVelocity[vesselCoords[0],vesselCoords[1]])
                
            pidx = np.where(vessel_velocities == max(vessel_velocities))
             
            V_cardiac_cycle[idx,0] = vesselCoords[0][pidx[0][0]]
            V_cardiac_cycle[idx,1] = vesselCoords[1][pidx[0][0]]
            V_cardiac_cycle[idx,2] = idx + 1
            
            V_cardiac_cycle[idx,3:V_cardiac_cycle.shape[1]] = self._correctedVelocityFrames[:,vesselCoords[0][pidx[0][0]],vesselCoords[1][pidx[0][0]]].ravel()
                    
        V_cardiac_cycle = abs(V_cardiac_cycle)
                    
        VmeanPerVesselList = np.zeros((V_cardiac_cycle.shape[0],1))
        MeanCurveOverAllVessels = np.zeros((1,self._correctedVelocityFrames.shape[0]))
        
        NormMeanCurvePerVessel = np.zeros((V_cardiac_cycle.shape[0],self._correctedVelocityFrames.shape[0]))
        normMeanCurveOverAllVessels = np.zeros((1,self._correctedVelocityFrames.shape[0]))
   
        for i in range(0,V_cardiac_cycle.shape[0]):
            
           VmeanPerVesselList[i,0:V_cardiac_cycle.shape[0]] = np.mean(V_cardiac_cycle[i,3:V_cardiac_cycle.shape[1]])
           MeanCurveOverAllVessels = MeanCurveOverAllVessels + np.squeeze((V_cardiac_cycle[i,3:V_cardiac_cycle.shape[1]]/(V_cardiac_cycle.shape[0])))
           
           NormMeanCurvePerVessel[i,0:self._correctedVelocityFrames.shape[0]] = V_cardiac_cycle[i,3:V_cardiac_cycle.shape[1]]/np.mean(V_cardiac_cycle[i,3:V_cardiac_cycle.shape[1]])
           
           # Velocity curves are first normalised and then averaged
           normMeanCurveOverAllVessels = normMeanCurveOverAllVessels + V_cardiac_cycle[i,3:V_cardiac_cycle.shape[1]]/np.mean(V_cardiac_cycle[i,3:V_cardiac_cycle.shape[1]])/(V_cardiac_cycle.shape[0])
             
        # Compute mean velocity  
        self._Vmean = np.mean(MeanCurveOverAllVessels)
        
        # Compute PI using normalised velocity curve of cardiac cycle averaged over all vessels
        self._PI_norm = (np.max(normMeanCurveOverAllVessels) - np.min(normMeanCurveOverAllVessels))/np.mean(normMeanCurveOverAllVessels)
        
        # Compute standard error of the mean of Vmean (adapted from MATLAB)
        allstdV = np.std(VmeanPerVesselList,ddof = 1)
        self._allsemV = allstdV/np.sqrt(V_cardiac_cycle.shape[0])
        
        # Compute standard error of the mean of PI_norm (adapted from MATLAB)
        allimaxV = np.where(normMeanCurveOverAllVessels == np.max(normMeanCurveOverAllVessels))[1]
        alliminV = np.where(normMeanCurveOverAllVessels == np.min(normMeanCurveOverAllVessels))[1]
        allstdnormV = np.std(NormMeanCurvePerVessel,ddof = 1,axis = 0)
        allstdmaxV = allstdnormV[allimaxV];
        allstdminV = allstdnormV[alliminV];
        allsemmaxV = allstdmaxV/np.sqrt(V_cardiac_cycle.shape[0])
        allsemminV = allstdminV/np.sqrt(V_cardiac_cycle.shape[0])
        allcovarmaxminV = 0
        self._allsemPI = np.sqrt(allsemmaxV**2 + allsemminV**2 - 2*allcovarmaxminV)[0]
              
    def _createVesselMask(self):
        """
        Iterates over the clusters found in _clusters and creates
        a mask of all the vessels.
        """
        
        mask = np.zeros(self._mask.shape,
                        dtype = np.int32)
        
        for labels in self._lone_vessels:
            mask += labels
        
        self._vesselMask        = mask.astype(bool)

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
            -Velocity per cycle
            
        Additional dictionary is created with following data per scan:
            - No. detected vessels
            - No. MPos vessels
            - No. MNeg vessels
            - No. MIso vessels
            - No. perpendicular vessels
            - No. non-perpendicular vessels
            - No. lone vessels
            - No. cluster vessels
            - Vmean lone vessels
            - Vmean standard error from mean (SEM)
            - PI_norm lone vessels
            - PI_norm SEM
            - No. BG mask pixels"""

        self._vesselDict = dict()
        self._velocityDict = dict()        
        
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
        total   = np.sum(np.asarray(self._lone_vessels))
        
        for idx, cluster in enumerate(self._lone_vessels):
         
            #Sort pixels in cluster by mean velocity (largest to smallest)
            pixels      = np.nonzero(cluster)
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
                    self._correctedVelocityFrames[:,x,y])),4)
                value_dict['maxV']          = round(np.max(np.abs(
                    self._correctedVelocityFrames[:,x,y])), 4)
                value_dict['PI']            = abs(round(div0(
                                             [(value_dict['maxV'] -
                                              value_dict['minV'])],
                                              value_dict['meanV'])[0],
                                                    4))
                value_dict['nPha']    = self._correctedVelocityFrames.shape[0]
                value_dict['imBlob']  = int(iMblob[x,y])
                
                
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
                
        'Additional dictionary is created below'
                
        velocity_dict = dict()
        velocity_dict['No. detected vessels']           = len(self._clusters)
        velocity_dict['No. MPos vessels']               = self._NoMPosClusters
        velocity_dict['No. MNeg vessels']               = self._NoMNegClusters
        velocity_dict['No. MIso vessels']               = self._NoMIsoClusters
        
        if self._readFromSettings('removeNonPerp'):
            
            velocity_dict['No. perpendicular vessels']      = self._Noperp_clusters
            velocity_dict['No. non-perpendicular vessels']  = len(self._non_perp_clusters)
            
        if self._readFromSettings('deduplicate'):
            
            velocity_dict['No. lone vessels']               = len(self._lone_vessels)
            velocity_dict['No. cluster vessels']            = len(self._cluster_vessels)
            velocity_dict['Vmean lone vessels']             = round(self._Vmean, 4)
            velocity_dict['PI_norm lone vessels']           = round(self._PI_norm, 4)
            
        else:
            
            velocity_dict['No. of vessels']            = len(self._lone_vessels)
            velocity_dict['Vmean vessels']             = round(self._Vmean, 4)
            velocity_dict['PI_norm vessels']           = round(self._PI_norm, 4)

        velocity_dict['Vmean SEM']                      = round(self._allsemV, 4)
        velocity_dict['PI_norm SEM']                    = round(self._allsemPI, 4)
        velocity_dict['No. BG mask pixels']             = sum(sum(self._mask == 1))
      
        self._velocityDict[0] = velocity_dict
        
        self._signalObject.setProgressBarSignal.emit(100)
        
    def _makeBatchAnalysisDict(self):
        """"Makes a dictionary containing the following statistics per scan:
            No. of vessels
            V_mean
            V_mean SEM
            PI_mean
            PI_mean SEM
            mean Velocity Trace
        """
        
        self._batchAnalysisDict = dict()
        
        if self._readFromSettings('deduplicate'):
        
            self._batchAnalysisDict['No. of vessels'] = self._velocityDict[0]['No. lone vessels'] 
            self._batchAnalysisDict['V_mean'] = self._velocityDict[0]['Vmean lone vessels'] 
            self._batchAnalysisDict['PI_mean'] = self._velocityDict[0]['PI_norm lone vessels']
            
        else:
            
            self._batchAnalysisDict['No. of vessels'] = self._velocityDict[0]['No. of vessels'] 
            self._batchAnalysisDict['V_mean'] = self._velocityDict[0]['Vmean vessels'] 
            self._batchAnalysisDict['PI_mean'] = self._velocityDict[0]['PI_norm vessels']
                
        self._batchAnalysisDict['V_mean SEM'] = self._velocityDict[0]['Vmean SEM'] 
        self._batchAnalysisDict['PI_mean SEM'] = self._velocityDict[0]['PI_norm SEM']        

        velocityTrace = np.zeros((self._batchAnalysisDict['No. of vessels'],
                                  len(self._correctedVelocityFrames)))
                
        for blob in range(1, self._batchAnalysisDict['No. of vessels'] + 1):
            
            for vessel in range(0,len(self._vesselDict)):
                
                if self._vesselDict[vessel]['iblob'] == blob and self._vesselDict[vessel]['ipixel'] == 1:

                    for num in range(1,len(self._correctedVelocityFrames) + 1):
                       
                       if num < 10:
                               
                           numStr = '0' + str(num)
                               
                       else:
                               
                           numStr = str(num)
                           
                       velocityTrace[blob - 1,num - 1] = self._vesselDict[vessel]['Vpha' + numStr]
                    
                    break

        self._batchAnalysisDict['Velocity trace'] = np.mean(velocityTrace,axis=0)

    def _writeToFile(self):
        """
        Creates a filename for the output and passes it to writeVesselDict
        along with the vesselDict object to be written. The velocityDict 
        object is written to a different file. 
        """

        #Message if no vessels were found
        if len(np.nonzero(self._lone_vessels)[0]) == 0:
            self._signalObject.errorMessageSignal.emit("No vessels Found")
            return
        
        #Get filename for textfile output for vesselData
        fname = self._dcmFilename[:-4]
        fname += "-Vessel_Data.txt"
        
        #Get filename for textfile output for velocityData
        fname_vel = self._dcmFilename[:-4]
        fname_vel += "-averagePIandVelocity_Data.txt"
        
        addonDict = self.getAddonDict()
        
        SELMADataIO.writeVesselDict(self._vesselDict,
                                    addonDict,
                                    fname)
        
        SELMADataIO.writeVelocityDict(self._velocityDict,
                                    addonDict,
                                    fname_vel)
        
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
    
        
    
        
        
        
        