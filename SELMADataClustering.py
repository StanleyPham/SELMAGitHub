# -*- coding: utf-8 -*-
"""
This fucntion belongs to the SELMAData module. Clustering of the flow and
magnitude voxels takes this place in this module and is separated from the 
main SELMAData module for clarity

@author: spham2
"""

# ====================================================================
import numpy as np
from skimage import measure 
import cv2

from PyQt5 import QtCore

# ====================================================================

import SELMAGUISettings

# -------------------------------------------------------------
'''Auxillary functions, used in the vessel analysis'''

def intersection(arrA, arrB):
    """ Checks for intersection between two 2D arrays"""
    #from: https://stackoverflow.com/questions/24477270/
    # python-intersection-of-2d-numpy-arrays

    return not set(map(tuple, arrA)).isdisjoint(map(tuple, arrB))

def remove_ccs_from_mask(entry_mask,conditional_mask):
    """This function is based on the version found in the original MATLAB
    implementation (line 1745 in PulsateGUI). This function finds the overlap
    between the flow mask and the significant magnitude mask. Coordinates of
    overlapping clusters are identified and the entire overlapping cluster is
    removed from the flow mask. The remaining flow mask is returned for the
    next magnitude direction."""
    
    #identify clusters in flow mask
    entry_ncomp, entry_mask_labels = cv2.connectedComponents(entry_mask) 
    
    if entry_ncomp == 1:
        
        output_mask = entry_mask #output entry mask directly
        
        return output_mask
        
    entry_mask_stats = measure.regionprops_table(entry_mask_labels,properties
                                                 = ('label','coords'))
    #extract coordinates of all pixels belonging in clusters
    entry_blob_coords = entry_mask_stats['coords'] 
    
    #skip if there is no overlap between flow and significant magnitude mask
    if not any(map(len,np.nonzero(entry_mask*conditional_mask))):
        
        output_mask = entry_mask #output entry mask directly
        
        return output_mask
    
    # extract pixel coordinates of overlapping clusters
    overlap_mask_stats = measure.regionprops_table((entry_mask 
                                                    * conditional_mask)
                                                   .astype(np.uint8)
                                                   ,properties=('label'
                                                                ,'coords'))
    
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
        
            output_mask[entry_blob_coords[blob][:,0],entry_blob_coords[blob]
                        [:,1]] = 1
            
    return output_mask

        
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

def clusterVessels(self):
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
    
    BasalGanglia           = self._readFromSettings('BasalGanglia')
    SemiovalCentre         = self._readFromSettings('SemiovalCentre')
    
    if BasalGanglia:
        
        PositiveMagnitude = 1
        NegativeMagnitude = 0
        IsointenseMagnitude = 0
        
        PositiveFlow = 1
        NegativeFlow = 0
        
    elif SemiovalCentre:
            
        PositiveMagnitude = 1
        NegativeMagnitude = 1
        IsointenseMagnitude = 1
        
        PositiveFlow = 0
        NegativeFlow = 1
  
    AdvancedClustering = self._readFromSettings('AdvancedClustering')
    
    if AdvancedClustering:
        
        PositiveMagnitude = self._readFromSettings('PositiveMagnitude')
        NegativeMagnitude = self._readFromSettings('NegativeMagnitude')
        IsointenseMagnitude = self._readFromSettings('IsointenseMagnitude')
        
        PositiveFlow = self._readFromSettings('PositiveFlow')
        NegativeFlow = self._readFromSettings('NegativeFlow')
    
    self._nComp     = 0
    self._clusters  = []

    'Positive magnitude clustering'

    # original flow masks with significant magnitudes in either direction
    entry_mask_VNeg = self._sigFlowNeg 
    entry_mask_VPos = self._sigFlowPos

    # Mask containing positive or negative flow with positive magnitude
    VNegMPos      = entry_mask_VNeg.astype(np.uint8) * self._sigMagPos
    VPosMPos      = entry_mask_VPos.astype(np.uint8) * self._sigMagPos
    
    if PositiveMagnitude and NegativeFlow:
    
        #VNegMPos
        # Find clusters with negative flow and postive magnitude
        ncomp_VNegMPos, labels = cv2.connectedComponents(VNegMPos
                                                         .astype(np.uint8))
        
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VNegMPos):
            self._clusters.append(labels == comp)
            
    else:
        
        ncomp_VNegMPos = 1
    
    # Remove found and overlapping clusters from flow mask
    output_mask_VNeg = remove_ccs_from_mask(entry_mask_VNeg,VNegMPos)
    
    if PositiveMagnitude and PositiveFlow:
    
        #VPosMPos
        # Find clusters with positive flow and postive magnitude
        ncomp_VPosMPos, labels = cv2.connectedComponents(VPosMPos.
                                                         astype(np.uint8))
        
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VPosMPos):
            self._clusters.append(labels == comp)
            
    else:
        
        ncomp_VPosMPos = 1
            
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
    
    if NegativeMagnitude and NegativeFlow:
    
        #VNegMNeg
        # Find clusters with negative flow and negative magnitude
        ncomp_VNegMNeg, labels = cv2.connectedComponents(VNegMNeg.
                                                         astype(np.uint8))
       
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VNegMNeg):
            self._clusters.append(labels == comp)
            
    else:
        
        ncomp_VNegMNeg = 1
    
    # Remove found and overlapping clusters from flow mask
    output_mask_VNeg = remove_ccs_from_mask(entry_mask_VNeg,VNegMNeg)
    
    if NegativeMagnitude and PositiveFlow:
    
        #VPosMNeg
        # Find clusters with positive flow and negative magnitude
        ncomp_VPosMNeg, labels = cv2.connectedComponents(VPosMNeg.
                                                         astype(np.uint8))
 
        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VPosMNeg):
            self._clusters.append(labels == comp)
            
    else:
        
        ncomp_VPosMNeg = 1
     
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
    
    if IsointenseMagnitude and NegativeFlow:
    
        #VNegMIso
        # Find clusters with negative flow and iosintense magnitude
        ncomp_VNegMIso, labels = cv2.connectedComponents(VNegMIso.
                                                         astype(np.uint8))

        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VNegMIso):
            self._clusters.append(labels == comp)
            
    else: 
        
        ncomp_VNegMIso = 1
        
    # Remove found and overlapping clusters from flow mask    
    output_mask_VNeg = remove_ccs_from_mask(entry_mask_VNeg,VNegMIso)
    
    if IsointenseMagnitude and PositiveFlow:
    
        #VPosMIso
        # Find clusters with positive flow and iosintense magnitude
        ncomp_VPosMIso, labels = cv2.connectedComponents(VPosMIso.
                                                         astype(np.uint8))

        # Append the found clusters to the total amount of found vessels
        for comp in range(1,ncomp_VPosMIso):
            self._clusters.append(labels == comp)
            
    else: 
        
        ncomp_VPosMIso = 1
    
    # Remove found and overlapping clusters from flow mask                            
    output_mask_VPos = remove_ccs_from_mask(entry_mask_VPos,VPosMIso)
    
    self._NoMIsoClusters = (ncomp_VNegMIso - 1) + (ncomp_VPosMIso - 1)
    
    """Previous attempt for clustering based solely on magnitude"""
    
    # 'Positive magnitude clustering'
    
    # if SemiovalCentre:
    
    #     # original flow masks with significant magnitudes in either direction
    #     entry_mask_VNeg = self._sigFlowNeg 
    #     entry_mask_VPos = self._sigFlowPos
    
    #     # Mask containing positive or negative flow with positive magnitude
    #     VNegMPos      = entry_mask_VNeg.astype(np.uint8) * self._sigMagPos
    #     VPosMPos      = entry_mask_VPos.astype(np.uint8) * self._sigMagPos
    
    #     if PositiveMagnitude and NegativeFlow:
    
    #         #VNegMPos
    #         # Find clusters with negative flow and postive magnitude
    #         ncomp_VNegMPos, labels = cv2.connectedComponents(VNegMPos.astype(np.uint8))
        
    #         # Append the found clusters to the total amount of found vessels
    #         for comp in range(1,ncomp_VNegMPos):
    #             self._clusters.append(labels == comp)
            
    #     else:
        
    #         ncomp_VNegMPos = 1
    
    #     # Remove found and overlapping clusters from flow mask
    #     output_mask_VNeg = remove_ccs_from_mask(entry_mask_VNeg,VNegMPos)
    
    #     if PositiveMagnitude and PositiveFlow:
    
    #         #VPosMPos
    #         # Find clusters with positive flow and postive magnitude
    #         ncomp_VPosMPos, labels = cv2.connectedComponents(VPosMPos.astype(np.uint8))
        
    #         # Append the found clusters to the total amount of found vessels
    #         for comp in range(1,ncomp_VPosMPos):
    #             self._clusters.append(labels == comp)
            
    #     else:
        
    #         ncomp_VPosMPos = 1
    
    #     # Remove found and overlapping clusters from flow mask
    #     output_mask_VPos = remove_ccs_from_mask(entry_mask_VPos,VPosMPos)
        
    #     self._NoMPosClusters = (ncomp_VNegMPos - 1) + (ncomp_VPosMPos - 1)
    
    # elif BasalGanglia:
    
    #     mask = self._mask.astype(bool)
    
    #     #VNegMPos
    #     # Find clusters with postive magnitude
    #     ncomp_MPos, labels = cv2.connectedComponents(self._sigMagPos.astype(np.uint8) * mask)
    
    #     if PositiveMagnitude: 
    
    #         # Append the found clusters to the total amount of found vessels
    #         for comp in range(1,ncomp_MPos):
    #             self._clusters.append(labels == comp)
            
    #         self._NoMPosClusters = (ncomp_MPos - 1)
            
    #     else:
        
    #         self._NoMPosClusters = 0
     
    # 'Negative magnitude clustering'
    
    # if SemiovalCentre:
    
    #     # Output masks of positive magnitude clustering is input for the
    #     # negative magnitude clustering. Flow masks with only negative 
    #     # significant magnitudes are remaining    
    #     entry_mask_VNeg = output_mask_VNeg.astype(np.uint8)
    #     entry_mask_VPos = output_mask_VPos.astype(np.uint8)
    
    #     # Mask containing positive or negative flow with negative magnitude
    #     VNegMNeg      = entry_mask_VNeg.astype(np.uint8) * self._sigMagNeg
    #     VPosMNeg      = entry_mask_VPos.astype(np.uint8) * self._sigMagNeg
    
    #     if NegativeMagnitude and NegativeFlow:
    
    #         #VNegMNeg
    #         # Find clusters with negative flow and negative magnitude
    #         ncomp_VNegMNeg, labels = cv2.connectedComponents(VNegMNeg.astype(np.uint8))
       
    #         # Append the found clusters to the total amount of found vessels
    #         for comp in range(1,ncomp_VNegMNeg):
    #             self._clusters.append(labels == comp)
            
    #     else:
        
    #         ncomp_VNegMNeg = 1
    
    #     # Remove found and overlapping clusters from flow mask
    #     output_mask_VNeg = remove_ccs_from_mask(entry_mask_VNeg,VNegMNeg)
    
    #     if NegativeMagnitude and PositiveFlow:
    
    #         #VPosMNeg
    #         # Find clusters with positive flow and negative magnitude
    #         ncomp_VPosMNeg, labels = cv2.connectedComponents(VPosMNeg.astype(np.uint8))
     
    #         # Append the found clusters to the total amount of found vessels
    #         for comp in range(1,ncomp_VPosMNeg):
    #             self._clusters.append(labels == comp)
            
    #     else:
        
    #         ncomp_VPosMNeg = 1
     
    #     # Remove found and overlapping clusters from flow mask
    #     output_mask_VPos = remove_ccs_from_mask(entry_mask_VPos,VPosMNeg)
    
    #     self._NoMNegClusters = (ncomp_VNegMNeg - 1) + (ncomp_VPosMNeg - 1)
    
    # elif BasalGanglia:
    
    #     #VNegMPos
    #     # Find clusters with negative magnitude
    #     ncomp_MNeg, labels = cv2.connectedComponents(self._sigMagNeg.astype(np.uint8) * mask)
    
    #     if NegativeMagnitude: 
     
    #         # Append the found clusters to the total amount of found vessels
    #         for comp in range(1,ncomp_MNeg):
    #             self._clusters.append(labels == comp)
            
    #         self._NoMNegClusters = (ncomp_MNeg - 1)
            
    #     else:
        
    #         self._NoMNegClusters = 0
        
    # 'Isointense magnitude clustering'
    
    # if SemiovalCentre:
    
    #     # Output masks of negative magnitude clustering is input for the
    #     # isointense magnitude clustering. Flow masks with neither positive or 
    #     # negative significant magnitude are remaining.
    #     entry_mask_VNeg = output_mask_VNeg.astype(np.uint8)
    #     entry_mask_VPos = output_mask_VPos.astype(np.uint8)
    
    #     # Mask containing positive or negative flow with isointense magnitude
    #     VNegMIso      = entry_mask_VNeg.astype(np.uint8) * self._sigMagIso
    #     VPosMIso      = entry_mask_VPos.astype(np.uint8) * self._sigMagIso
    
    #     if IsointenseMagnitude and NegativeFlow:
    
    #         #VNegMIso
    #         # Find clusters with negative flow and iosintense magnitude
    #         ncomp_VNegMIso, labels = cv2.connectedComponents(VNegMIso.astype(np.uint8))
    
    #         # Append the found clusters to the total amount of found vessels
    #         for comp in range(1,ncomp_VNegMIso):
    #             self._clusters.append(labels == comp)
            
    #     else: 
        
    #         ncomp_VNegMIso = 1
        
    #     # Remove found and overlapping clusters from flow mask    
    #     output_mask_VNeg = remove_ccs_from_mask(entry_mask_VNeg,VNegMIso)
    
    #     if IsointenseMagnitude and PositiveFlow:
    
    #         #VPosMIso
    #         # Find clusters with positive flow and iosintense magnitude
    #         ncomp_VPosMIso, labels = cv2.connectedComponents(VPosMIso.astype(np.uint8))
    
    #         # Append the found clusters to the total amount of found vessels
    #         for comp in range(1,ncomp_VPosMIso):
    #             self._clusters.append(labels == comp)
            
    #     else: 
        
    #         ncomp_VPosMIso = 1
    
    #     # Remove found and overlapping clusters from flow mask                            
    #     output_mask_VPos = remove_ccs_from_mask(entry_mask_VPos,VPosMIso)
    
    #     self._NoMIsoClusters = (ncomp_VNegMIso - 1) + (ncomp_VPosMIso - 1)
    
    # elif BasalGanglia:
    
    #     #VNegMPos
    #     # Find clusters with isointense magnitude
    #     ncomp_MIso, labels = cv2.connectedComponents(self._sigMagIso.astype(np.uint8) * mask)
    
    #     if IsointenseMagnitude: 
    
    #         # Append the found clusters to the total amount of found vessels
    #         for comp in range(1,ncomp_MIso):
    #             self._clusters.append(labels == comp)
            
    #         self._NoMIsoClusters = (ncomp_MIso - 1)
            
    #     else:
        
    #         self._NoMIsoClusters = 0
    
    #Cluster only significant magnitude, to determine iMblob
    NclusPos, self._posMagClusters     = cv2.connectedComponents(
                                self._sigMagPos * self._mask)
    NclusNeg, self._negMagClusters     = cv2.connectedComponents(
                                self._sigMagNeg * self._mask)