#!/usr/bin/env python

"""
This module contains the main function of the SELMA program.

"""

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

    COMPANY, APPNAME, _ = SELMAGUISettings.getInfo()

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
    #TODO: all slots should be in SGM or SDM
    # ----------------------------------------
    #Signals from mainwindow (menubar)
    SGM.mainWin.loadMaskSignal      .connect(SDM.loadMaskSlot)
    SGM.mainWin.segmentMaskSignal   .connect(SDM.segmentMaskSlot)
    SGM.mainWin.openFileSignal      .connect(SDM.loadDCMSlot)
    SGM.mainWin.openClassicSignal   .connect(SDM.loadClassicDCMSlot)
    SGM.mainWin.openT1Signal        .connect(SDM.loadT1DCMSlot)
    SGM.mainWin.analyseVesselSignal .connect(SDM.analyseVesselSlot)
    SGM.mainWin.analyseBatchSignal  .connect(SDM.analyseBatchSlot)
    SGM.mainWin.switchViewSignal    .connect(SDM.switchViewSlot)
    SGM.mainWin.applyMaskSignal     .connect(SDM.applyMaskSlot)
    SGM.mainWin.saveMaskSignal      .connect(SDM.saveMaskSlot)
    
    #Signals from mouseEvents
    SGM.mainWin._imageViewer._scene.mouseMove.connect(SDM.pixelValueSlot)
    SGM.mainWin._imageViewer._view.wheelEventSignal.connect(SDM.newFrameSlot)
    
    #Signals from ImVar
    SGM.mainWin.signalObj.getVarSignal.connect(
            SDM.getVarSlot)
    SGM.mainWin.signalObj.setVarSignal.connect(
            SDM.setVarSlot)
    
    
    #Signals from processing
    SDM.signalObject.sendVesselMaskSignal.connect(SGM.setVesselMaskSlot)
    SDM.signalObject.setPixmapSignal.connect(SGM.setPixmapSlot)
    SDM.signalObject.setProgressBarSignal.connect(
            SGM.setProgressBarSlot)
    SDM.signalObject.setFrameCountSignal.connect(SGM.setFrameCounterSlot)
    SDM.signalObject.sendMaskSignal.connect(SGM.setMaskSlot)
    SDM.signalObject.pixelValueSignal.connect(
            SGM.mainWin._imageViewer.mouseHover)
    SDM.signalObject.errorMessageSignal.connect(
            SGM.mainWin.errorMessageSlot)
    SDM.signalObject.sendImVarSignal.connect(
            SGM.listenForVarsSlot)
    SDM.signalObject.setProgressLabelSignal.connect(
            SGM.setProgressLabelSlot)
    

    # ---------------------------------------
    sys.exit(app.exec_())
    


if __name__ == '__main__':
    main()