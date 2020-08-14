#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`SELMADicom`

"""


# ====================================================================
#IO
import SELMADicom
import pydicom
import numpy as np

# ====================================================================

class SELMAClassicDicom(SELMADicom.SELMADicom):
    """
    This class contains all methods concerning the handling of a classic 
    .dcm (Dicom) file. All the manufacturer specific information, such as keys,
    addresses etc. is managed here.
    
    This class assumes the following:
        the manufacturer and velocity encoding value are the same for 
        all files in the directory.
    """
    
    def __init__(self, dcmFilenames):
        """Read the dicom header using pydicom. 
        Also extract the pixel array.
        Call the functions that initiate the Dicom."""
        
        self._dcmFilenames    = dcmFilenames
        
        self._tags              = dict()
        self._DCMs              = list()
        self._numFrames         = len(self._dcmFilenames)
        
        # load the dicoms
        #Iterate over the dicom files in the directory.
        rawFrames           = []
        for filename in self._dcmFilenames:
            DCM             = pydicom.dcmread(filename)
            self._DCMs.append(DCM)
            rawFrames.append(DCM.pixel_array)
            
        self._rawFrames     = np.asarray(rawFrames)
            
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
        
        #Construct velocity frames if necessary
        self._makeVelocityFrames()
        
    #Setter functions
    # ------------------------------------------------------------------    
    
    def setVenc(self, venc):
        self._tags['venc'] = venc
        if self._tags['manufacturer'] == 'SIEMENS':
             self._makeVelocityFrames()
    
    '''Private'''
    
    def _findManufacturer(self):
        """Extract the manufacturer from the dicom. It's assumed that every
        dicom file in the list has the same manufacturer.
        """
        self._tags['manufacturer'] = self._DCMs[0][0x0008, 0x0070].value
    
    def _findRescaleValues(self):
        """Finds the rescale slope and intercept
        and applies it to the frames"""
        
        rescaleSlopes     = []
        rescaleIntercepts = []
        
        
        #Philips
        if self._tags['manufacturer'] == 'Philips Medical Systems':
            dcmRescaleInterceptAddress  = 0x2005, 0x100D
            dcmRescaleSlopeAddress      = 0x2005, 0x100E
            
            
            for i in range(self._numFrames):
                rescaleSlope        = float(self._DCMs[i]
                                        [dcmRescaleSlopeAddress].value)
                rescaleIntercept    = float(self._DCMs[i]
                                        [dcmRescaleInterceptAddress].value)
                
                rescaleSlopes.append(rescaleSlope)
                rescaleIntercepts.append(rescaleIntercept)


        elif self._tags['manufacturer'] == 'SIEMENS':
            #Try to find the rescale values in the slices. If not available,
            #???
            #Also try to calculate the value for venc
            
            dcmRescaleInterceptAddress  = 0x0028, 0x1052
            dcmRescaleSlopeAddress      = 0x0028, 0x1053

            for i in range(self._numFrames):
                try:
                    rescaleSlope        = float(self._DCMs[i]
                                        [dcmRescaleSlopeAddress].value)
                    rescaleIntercept    = float(self._DCMs[i]
                                        [dcmRescaleInterceptAddress].value)
                    rescaleSlopes.append(rescaleSlope)
                    rescaleIntercepts.append(rescaleIntercept)
                    
                    #Set the venc based on the rescale slope of the phase 
                    #images
#                    try:
#                        self._tags['venc']
#                    
#                    except:
#                        #TODO: Check if right
#                        if self._DCMs[i][0x8, 0x8].value[2] == 'V':
#                            self._tags['venc'] = rescaleSlope
                    
                except:
                    rescaleSlopes.append([])  
                    rescaleIntercepts.append([])
                    


         #Other manufacturers
        #
        #
        #
    
        
        self._tags['rescaleSlopes']     = rescaleSlopes
        self._tags['rescaleIntercepts'] = rescaleIntercepts



    def _findVEncoding(self):
        """Gets the velocity encoding maximum in the z-direction from the DCM.
        It's assumed that this is constant for all frames."""
        
        try:
            venc    = self._tags['venc']
        except:        
            venc = 1            #Default value, if nothing can be found.
        
        #Philips        
        if self._tags['manufacturer'] == 'Philips Medical Systems':
            vencAddress                 = 0x2001, 0x101A
            venc                        = self._DCMs[0][vencAddress].value
            venc                        = venc[-1] 
        
        
        #GE
        if self._tags['manufacturer'] == 'GE MEDICAL SYSTEMS':
            vencAddress                 = 0x0019, 0x10CC
            venc                        = self._DCMs[0][vencAddress].value
            #Change from mm/s to cm/s
            venc                        = venc[-1] / 10
        
        
        #Siemens
        #possible location:         0x0019, 0x1012
        #possible location:         0x0019, 0x1013
        
        
        #Other manufacturers
                
        self._tags['venc'] = venc
            
            
            
    def _findFrameTypes(self):
        """Find the frame types per manufacturer.
        Method differs for each manifacturer."""
        
        self._tags['frameTypes'] = []
        
        #Philips
        if self._tags['manufacturer'] == 'Philips Medical Systems':
            dcmImageTypeAddress         = 0x0008, 0x0008
            
            for i in range(self._numFrames):
                frameType = self._DCMs[i][dcmImageTypeAddress].value[2]
                self._tags['frameTypes'].append(frameType)
                
        #Siemens
        elif self._tags['manufacturer'] == 'SIEMENS':
            dcmImageTypeAddress         = 0x0008, 0x0008
            
            for i in range(self._numFrames):
                frameType = self._DCMs[i][dcmImageTypeAddress].value[2]
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
        if self._tags['manufacturer'] == 'Philips Medical Systems':
            self._tags['targets']['phase']      = 'PHASE CONTRAST M'
            self._tags['targets']['velocity']   = 'VELOCITY MAP'
            self._tags['targets']['magnitude']  = "M_FFE"
            self._tags['targets']['modulus']    = "M_PCA"
            
        
        #Siemens
        elif self._tags['manufacturer'] == 'SIEMENS':
            self._tags['targets']['phase']      = 'P'
            self._tags['targets']['velocity']   = 'V'
            self._tags['targets']['magnitude']  = "MAG"
            self._tags['targets']['modulus']    = "M"
        
    
    def _rescaleFrames(self):
        ''' Applies the rescale slope and intercept to the frames. '''
        
        self._rescaledFrames = []
        for i in range(len(self._rawFrames)):
            rescaleSlope        = self._tags['rescaleSlopes'][i]
            rescaleIntercept    = self._tags['rescaleIntercepts'][i]
            rawFrame            = self._rawFrames[i]
            
            #Skip the slices without slope or intercept
            if rescaleSlope == [] or rescaleIntercept == []:
                rescaledFrame   = rawFrame
            else:
                rescaledFrame   = (rawFrame - rescaleIntercept)/rescaleSlope
            
            self._rescaledFrames.append(rescaledFrame)
    
    
    def _makeVelocityFrames(self):
        '''
        Construct velocity frames out of the phase frames if any phase frames
        exist. Formula: v = phase * venc / pi
        '''
        if len(self._phaseFrames) > 0:
            
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
            
        