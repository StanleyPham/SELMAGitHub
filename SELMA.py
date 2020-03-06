#!/usr/bin/env python

"""
This module contains the main function of the SELMA program.

"""

# ====================================================================

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#from future_builtins import *

# ====================================================================

import sys
from PyQt5 import (QtCore, QtGui, QtWidgets)

# ====================================================================

import SELMADataModels
import SELMAGUIModels
import SELMAGUISettings

# ====================================================================

def main():
    """ SELMA - [Abbreviation] """

    COMPANY = "UMCu"
    APPNAME = "SELMA"

    app = QtWidgets.QApplication(sys.argv)
    
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    app.setOrganizationName(COMPANY)
    app.setApplicationName(APPNAME)
    app.setWindowIcon(QtGui.QIcon("SELMA.png"))

    #Model classes
    SGM = SELMAGUIModels.SelmaGUIModel(APPNAME = APPNAME)
    SDM = SELMADataModels.SelmaDataModel()
    
    #Initialise settings
    settings = SELMAGUISettings.SelmaSettings()
    settings.applySettings()
    
    #Connect signals
    # ----------------------------------------
    #Signals from mainwindow (menubar)
    SGM.mainWin.loadMaskSignal.connect(SDM.loadMaskSlot)
    SGM.mainWin.segmentMaskSignal.connect(SDM.segmentMaskSlot)
    SGM.mainWin.openFileSignal.connect(SDM.loadDCMSlot)
    SGM.mainWin.openDirSignal.connect(SDM.loadDCMDirSlot)
    SGM.mainWin.analyseVesselSignal.connect(SDM.analyseVesselSlot)
    SGM.mainWin.analyseBatchSignal.connect(SDM.analyseBatchSlot)
    SGM.mainWin.applyMaskSignal.connect(SDM.applyMaskSlot)
    SGM.mainWin.saveMaskSignal.connect(SDM.saveMaskSlot)
    
    #Signals from mouseEvents
    SGM.mainWin._imageViewer._scene.mouseMove.connect(SDM.pixelValueSlot)
    SGM.mainWin._imageViewer._view.wheelEventSignal.connect(SDM.newFrameSlot)
    
    #Signals from processing
    SDM.signalObject.sendVesselMaskSignal.connect(SGM.setVesselMaskSlot)
    SDM.signalObject.setPixmapSignal.connect(SGM.setPixmapSlot)
    SDM.signalObject.setProgressBarSignal.connect(
            SGM.mainWin._imageViewer.setProgressBar)
    SDM.signalObject.setFrameCountSignal.connect(SGM.setFrameCounterSlot)
    SDM.signalObject.sendMaskSignal.connect(SGM.setMaskSlot)
    SDM.signalObject.pixelValueSignal.connect(
            SGM.mainWin._imageViewer.mouseHover)
    SDM.signalObject.errorMessageSignal.connect(
            SGM.mainWin.errorMessageSlot)

    # ---------------------------------------
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()