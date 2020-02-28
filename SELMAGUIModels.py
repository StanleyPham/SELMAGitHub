#!/usr/bin/env python

"""
This module is contains all the relevant classes that form the first layer 
between the SELMA GUI and the data objects. It contains the following classes:

+ :class:`DicomReader`
+ :class:`ReadMask`
    
"""

# ====================================================================

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#from future_builtins import *

# ====================================================================

import numpy as np
from PyQt5 import (QtCore, QtGui, QtWidgets, Qt)
#import qimage2ndarray

# ====================================================================

import SELMAGUI

# ====================================================================

class SelmaGUIModel:
    """Class containing a SELMAGUI as well as functions for 
    setting / reading its properties."""
    
    def __init__(self, APPNAME = "SELMA"):
        """:param QPixmap pixmap: |QPixmap| to display"""
    
        self.mainWin = SELMAGUI.SELMAMainWindow()
        self.mainWin.setWindowTitle(APPNAME)
        self.mainWin.readSettings()
        self.mainWin.show()
        
    
    '''Public'''
     
    #Signals
    # ------------------------------------------------------------------
    
    
    #Slots
    # ------------------------------------------------------------------
    def setPixmapSlot(self, frame):
        """Passes the pixmap to the mainWin."""
        self.mainWin.setPixmap(frame)
        
    def setFrameCounterSlot(self, frameCounter, maxFrames):
        """Passes the frame count to the mainWin."""
        self.mainWin.setFrameCounter(frameCounter, maxFrames)
    
    def setMaskSlot(self, mask):
        """Passes the mask to the mainWin."""
        self.mainWin.setMask(mask)
        
    def setVesselMaskSlot(self, mask):
        """Passes the vesselMask to the mainWin."""
        self.mainWin.setVesselMask(mask)
    
    #Getter functions
    # ------------------------------------------------------------------    
    
    #Setter functions
    # -----------------------------------------------------------------
    
    '''Auxillary'''
    
    
    