#!/usr/bin/env python

"""
This module is contains all the relevant classes that form the first layer 
between the SELMA GUI and the data objects. It contains the following classes:

+ :class:`SelmaGUIModel`
    
"""

# ====================================================================

import SELMAGUI

# ====================================================================

class SelmaGUIModel:
    """Class containing a SELMAGUI as well as functions for 
    setting / reading its properties."""
    
    def __init__(self, APPNAME = "SELMA"):
        """:param QPixmap pixmap: |QPixmap| to display"""
    
        self.mainWin = SELMAGUI.SELMAMainWindow() # reference to SELMAGUI.py and SELMAMainWindow class
        self.mainWin.setWindowTitle(APPNAME)
        self.mainWin.readSettings()
        self.mainWin.show()
        
    
    '''Public'''
     
    #Signals
    # ------------------------------------------------------------------
    
    
    #Slots
    # These slots are called in SELMA.py
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
        
    def listenForVarsSlot(self, variables):
        """passes the variables to mainwin.ImVar"""
        self.mainWin.imVarWindow.listenForVars(variables)
        
    def setProgressBarSlot(self, val):
        """Passes the progressbar value to mainwin"""
        self.mainWin.setProgressBar(val)
        
    def setProgressLabelSlot(self, text):
        """Passes the progress message to mainwin"""
        self.mainWin.setProgressLabel(text)
    
    #Getter functions
    # ------------------------------------------------------------------    
    
    #Setter functions
    # -----------------------------------------------------------------
    
    '''Auxillary'''
    
    
    