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
    
    '''Private'''
    
    def _findManufacturer(self):
        """Extract the manufacturer from the dicom"""
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
            #Calculate the encoding based on the venc and the min/max data.
            #TODO: try to find it in the private text tag.
            
            self._findVEncoding()
            venc    = self._tags['venc']

            for i in range(self._numFrames):
                maxVal  = self._DCMs[i][0x0028,0x0107].value
                rescaleSlopes.append(maxVal / (2 * venc))
                rescaleIntercepts.append(maxVal / 2)
            #...


         #Other manufacturers
        #
        #
        #
    
        
        self._tags['rescaleSlopes']     = rescaleSlopes
        self._tags['rescaleIntercepts'] = rescaleIntercepts



    def _findVEncoding(self):
        """Gets the velocity encoding maximum in the z-direction from the DCM.
        It's assumed that this is constant for all frames."""
        
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
                
                
            
        #Other manufacturers
        #
        #
        #
    
    
    
        