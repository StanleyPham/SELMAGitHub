#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`SELMADicom`

"""


# ====================================================================
#IO

import pydicom
import numpy as np

# ====================================================================

class SELMADicom:
    """
    This class contains all methods concerning the handling of .dcm (Dicom)
    data. All the manufacturer specific information, such as keys, addresses
    etc. is managed here.
    """
    
    def __init__(self, dcmFilename):
        """Read the dicom header using pydicom. 
        Also extract the pixel array.
        Call the functions that initiate the Dicom."""
        
        self._dcmFilename   = dcmFilename
        self._DCM           = pydicom.dcmread(self._dcmFilename)
        
        self._tags          = dict()
        self._rawFrames     = self._DCM.pixel_array
        self._numFrames     = len(self._rawFrames)
        
        #Get manufacturer
        self._findManufacturer()
        
        #find important Tags
        self._findRescaleValues()    
        self._findVEncoding()
        self._findFrameTypes()
        
        self._findTargets()
        
        #Get rescale values and apply
        self._rescaleFrames()
        
        #Sort the frames on their type
        self._orderFramesOnType()
        
        #Create velocity frames if necessary:
        self._makeVelocityFrames()
    
    '''Public'''
    #Getter functions
    # ------------------------------------------------------------------    
    def getTags(self):
        return self._tags
    
    def getFrames(self):
        return self._rescaledFrames
    
    def getRawFrames(self):
        return self._rawFrames
    
    def getDCM(self):
        return self._DCM
    
    def getNumFrames(self):
        return self._numFrames
    
    def getVelocityFrames(self):
        return self._velocityFrames
    
    def getMagnitudeFrames(self):
        return self._magnitudeFrames
    
    def getModulusFrames(self):
        return self._modulusFrames
    
    def getRawVelocityFrames(self):
        return self._rawVelocityFrames
    
    def getRawMagnitudeFrames(self):
        return self._rawMagnitudeFrames
    
    def getRawModulusFrames(self):
        return self._rawModulusFrames
    
    #Setter functions
    # ------------------------------------------------------------------    
    
    def setVenc(self, venc):
        self._tags['venc'] = venc
        
        
    '''Private'''
    # Setup data from .dcm header
    # ------------------------------------------------------------------    
    
    """Where to find the relevant dicom tags for different vendors.    
    
    Philips:
    Frame data stored in                                (5200, 9230) 
    Manufacturer stored in :                            (2005, 0014) 
    Manufacturer specific data stored in:               (2005, 140f)  
    Image Type stored in                                (0008, 0008)
    Image Types for the following frames:
        Modulus:    ['ORIGINAL', 'PRIMARY', 'M_PCA', 'M', 'PCA']
        Magnitude:  ['ORIGINAL', 'PRIMARY', 'M_FFE', 'M', 'FFE']
        Velocity:   ['ORIGINAL', 'PRIMARY', 'VELOCITY MAP', 'P', 'PCA']
    Rescale values stored in:
        Intercept:                                      (2005, 100d)
        Slope:                                          (2005, 100e)
    VEnc value stored at:
        vEncAddress                                     (0018, 9197)
        vEncMaxAddress                                  (0018, 9217)
    
    """
    
    def _findManufacturer(self):
        """Extract the manufacturer from the dicom"""
        self._tags['manufacturer'] = self._DCM[0x0008, 0x0070].value
    
    def _findRescaleValues(self):
        """Finds the rescale slope and intercept
        and applies it to the frames"""
        
        rescaleSlopes     = []
        rescaleIntercepts = []
        
        
        #Philips
        if 'philips' in self._tags['manufacturer'].lower():
            dcmFrameAddress             = 0x5200, 0x9230
            dcmPrivateCreatorAddress    = 0x2005, 0x140f
            dcmRescaleSlopeAddress      = 0x2005, 0x100E
            dcmRescaleInterceptAddress  = 0x2005, 0x100D
            
            
            for i in range(self._numFrames):
                rescaleSlope        = float(self._DCM[dcmFrameAddress][i]           
                                        [dcmPrivateCreatorAddress][0]      
                                        [dcmRescaleSlopeAddress].value)
                rescaleIntercept    = float(self._DCM[dcmFrameAddress][i]           
                                        [dcmPrivateCreatorAddress][0]      
                                        [dcmRescaleInterceptAddress].value)
                
                rescaleSlopes.append(rescaleSlope)
                rescaleIntercepts.append(rescaleIntercept)



         #Other manufacturers
        #
        #
        #
    
        
        self._tags['rescaleSlopes']     = rescaleSlopes
        self._tags['rescaleIntercepts'] = rescaleIntercepts



    def _findVEncoding(self):
        """Gets the velocity encoding maximum in the z-direction from the DCM.
        It's assumed that this is constant for all frames."""
        
        #Philips
        
        if 'philips' in self._tags['manufacturer'].lower():
            vencAddress                 = 0x2001, 0x101A
            venc                        = self._DCM[vencAddress].value
            venc                        = venc[-1] 
        
#        if self._tags['manufacturer'] == 'Philips Medical Systems':
#            dcmFrameAddress             = 0x5200, 0x9230
#            vEncAddress                 = 0x0018, 0x9197
#            vEncMaxAddress              = 0x0018, 0x9217
#            
#            try:
#                venc = self._DCM[dcmFrameAddress] [0]       \
#                                [vEncAddress]     [0]       \
#                                [vEncMaxAddress].value
#            except:
#                venc = 0
                
        #Other manufacturers
                
        self._tags['venc'] = venc
            
            
            
    def _findFrameTypes(self):
        """Find the frame types per manufacturer.
        Method differs for each manifacturer."""
        
        self._tags['frameTypes'] = []
        
        #Philips
        if 'philips' in self._tags['manufacturer'].lower():
            dcmFrameAddress             = 0x5200, 0x9230
            dcmPrivateCreatorAddress    = 0x2005, 0x140f
            dcmImageTypeAddress         = 0x0008, 0x0008
            
            for i in range(self._numFrames):
                frameType = self._DCM[dcmFrameAddress][i]                   \
                                [dcmPrivateCreatorAddress][0]               \
                                [dcmImageTypeAddress].value[2]
                self._tags['frameTypes'].append(frameType)
            
            
        #Other manufacturers
        #
        #
        #
        
        
    def _findTargets(self):
        """
        Saves the manufacturer specific names for the phase, velocity,
        magnutide and modulus frames.
        
        """
        self._tags['targets'] = dict()
        
        #Philips
        if 'philips' in self._tags['manufacturer'].lower():
            self._tags['targets']['phase']      = 'PHASE'
            self._tags['targets']['velocity']   = 'VELOCITY'
            self._tags['targets']['magnitude']  = "M_FFE"
            self._tags['targets']['modulus']    = "M_PCA"
            
        
        #Siemens
        if 'siemens' in self._tags['manufacturer'].lower():
            self._tags['targets']['phase']      = 'P'
            self._tags['targets']['velocity']   = 'V'
            self._tags['targets']['magnitude']  = "MAG"
            self._tags['targets']['modulus']    = "M"
            
            
    
    # Apply changes to the frames
    # ------------------------------------------------------------------    

    def _rescaleFrames(self):
        ''' Applies the rescale slope and intercept to the frames. '''
         
        self._rescaledFrames = []
        for i in range(len(self._rawFrames)):
            rescaleSlope        = self._tags['rescaleSlopes'][i]
            rescaleIntercept    = self._tags['rescaleIntercepts'][i]
            
            rawFrame            = self._rawFrames[i]
            rescaledFrame       = (rawFrame - rescaleIntercept)/rescaleSlope
            
            self._rescaledFrames.append(rescaledFrame)



    def _orderFramesOnType(self):
        """Uses the indices found in findFrameTypes to create an array for
        the magnitude, modulus, and velocity frames."""

        self._magnitudeFrames           = []
        self._rawMagnitudeFrames        = []
        self._modulusFrames             = []
        self._rawModulusFrames          = []
        self._velocityFrames            = []
        self._rawVelocityFrames         = []
        self._phaseFrames               = []
        self._rawPhaseFrames            = []
        
        frameTypes      = self._tags['frameTypes']
        targets         = self._tags['targets']
        
        for idx in range(self._numFrames):
                        
            if targets['velocity'] in frameTypes[idx]:
                self._velocityFrames.append(self._rescaledFrames[idx])
                self._rawVelocityFrames.append(self._rawFrames[idx])
                
            elif targets['magnitude'] in frameTypes[idx]:
                self._magnitudeFrames.append(self._rescaledFrames[idx])
                self._rawMagnitudeFrames.append(self._rawFrames[idx])
                
            elif targets['modulus'] in frameTypes[idx]:
                self._modulusFrames.append(self._rescaledFrames[idx])
                self._rawModulusFrames.append(self._rawFrames[idx])
                
            elif targets['phase'] in frameTypes[idx]:
                self._phaseFrames.append(self._rescaledFrames[idx])
                self._rawPhaseFrames.append(self._rawFrames[idx])
            
    
    def _makeVelocityFrames(self):
        '''
        Construct velocity frames out of the phase frames if any phase frames
        exist. Formula: v = phase * venc / pi
        '''
        if len(self._phaseFrames) > 0 and len(self._velocityFrames) == 0:
            
            venc = self._tags['venc']
            
            #Check if the velocity frames aren't accidentally stored as phase
            
            if np.round(np.max(self._phaseFrames), 1) == venc and \
               np.round(np.min(self._phaseFrames), 1) == -venc:
               
                self._velocityFrames        = self._phaseFrames
                self._rawVelocityFrames     = self._rawPhaseFrames
                return
            
            #Else, compute velocity frames from the phaseFrames
            for idx in range(len(self._phaseFrames)):
                phaseFrame  = self._phaseFrames[idx] * venc / np.pi
                rawPhaseFrame  = self._rawPhaseFrames[idx] * venc / np.pi
                self._velocityFrames.append(phaseFrame)
                self._rawVelocityFrames.append(rawPhaseFrame)
    
        