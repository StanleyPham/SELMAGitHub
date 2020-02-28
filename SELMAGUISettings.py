#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`ValueHoverLabel`
+ :class:`SynchableGraphicsView`
+ :class:`ImageViewer`

"""

# ====================================================================

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#from future_builtins import *

# ====================================================================

from PyQt5 import (QtCore, QtGui, QtWidgets)

# ====================================================================
class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        

class SelmaSettings(QtWidgets.QWidget):
    """
    This class contains the UI for interacting with the user settings. 
    Settings are viewed, edited and saved.
    
    The settings window displays the following tabs:
        General     - for the main functioning of the program
        Ghosting    - for the removeGhosting function
        Reset       - for resetting the settings to their default
    """
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        
        #Create window
        self.initGui()
        
        #Load settings from disk
        self.getSettings()
    
    def initGui(self):
        
        self.setGeometry(QtCore.QRect(100, 100, 300, 200))
        self.setWindowTitle("Settings")
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        
        #Add tabs
        self.tabs           = QtWidgets.QTabWidget()
        self.mainTab        = QtWidgets.QWidget()
        self.ghostingTab    = QtWidgets.QWidget()
        self.resetTab       = QtWidgets.QWidget()
        
        self.tabs.addTab(self.mainTab,      "General")
        self.tabs.addTab(self.ghostingTab,  "Ghosting")
        self.tabs.addTab(self.resetTab,     "Reset")
        
        #Design Tabs
        self.initMainTab()
        self.initGhostingTab()
        self.initResetTab()
        
        #Add to layout
        self.layout         = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.tabs)
        
        #Buttons
        self.okButton       = QtWidgets.QPushButton("OK")
        self.cancelButton   = QtWidgets.QPushButton("Cancel")
        self.okButton.pressed.connect(self.applySettings)
        self.cancelButton.pressed.connect(self.close)
        
        self.buttonLayout   = QtWidgets.QHBoxLayout(self)
        self.buttonLayout.addWidget(self.okButton)
        self.buttonLayout.addWidget(self.cancelButton)
        self.layout.addLayout(self.buttonLayout)
        
        #Error Label
        self.errorLabel     = QtWidgets.QLabel("")
        self.errorLabel.setStyleSheet("QLabel {color: red }")
        self.layout.addWidget(self.errorLabel)
        
        #Finish
        self.setLayout(self.layout)
        
    def initMainTab(self):
        """The tab containing the general settings.        """
        
        self.mainTab.lineEdit1  = QtWidgets.QLineEdit()
        self.mainTab.lineEdit2  = QtWidgets.QLineEdit()
        self.mainTab.lineEdit3  = QtWidgets.QLineEdit()
        self.mainTab.cBox1      = QtWidgets.QCheckBox()
        self.mainTab.cBox2      = QtWidgets.QCheckBox()
        
        self.mainTab.label1     = QtWidgets.QLabel("Median filter diameter")
        self.mainTab.label2     = QtWidgets.QLabel("Confindence interval")
        self.mainTab.label3     = QtWidgets.QLabel("White Matter Probability")
        self.mainTab.label4     = QtWidgets.QLabel("Average over cardiac cycle")
        self.mainTab.label5     = QtWidgets.QLabel("Exclude perpendicular vessels")
        
        self.mainTab.label1.setToolTip("Diameter of the kernel used in"             +
                                       " the median filtering operations."          +
                                       "\nDefault is 53.")
        self.mainTab.label2.setToolTip("Confidence interval used to determine"      + 
                                       " whether a vessel is significant."          +
                                       "\nDefault is 0.05.")
        self.mainTab.label3.setToolTip("Threshold for classifying a voxel"          +
                                       " as white matter. \nDefault is 0.5")
        self.mainTab.label4.setToolTip("When toggled on, the reported values"       +
                                       " per voxel are averaged over the \n"        +
                                       "cycle. If not on, the values are from"      +
                                       " the first phase.")
        self.mainTab.label5.setToolTip("When toggled on, the significant vessels"   +
                                       " are filtered for any vessels that lie \n"  +
                                       "perpendicular to the imaging plane.")
        

        self.mainTab.layout     = QtWidgets.QGridLayout()
        self.mainTab.layout.addWidget(self.mainTab.lineEdit1, 0,0)
        self.mainTab.layout.addWidget(self.mainTab.lineEdit2, 1,0)
        self.mainTab.layout.addWidget(self.mainTab.lineEdit3, 2,0)
        
        self.mainTab.layout.addWidget(QHLine(),               3,0,1,2)
        
        self.mainTab.layout.addWidget(self.mainTab.cBox1,     4,0)
        self.mainTab.layout.addWidget(self.mainTab.cBox2,     5,0)
        
        self.mainTab.layout.addWidget(self.mainTab.label1,    0,1)
        self.mainTab.layout.addWidget(self.mainTab.label2,    1,1)
        self.mainTab.layout.addWidget(self.mainTab.label3,    2,1)
        self.mainTab.layout.addWidget(self.mainTab.label4,    4,1)
        self.mainTab.layout.addWidget(self.mainTab.label5,    5,1)
        
        self.mainTab.setLayout(self.mainTab.layout)
        
    def initGhostingTab(self): 
        """The tab containing the removeGhosting settings."""

        #Toggle Ghosting
        self.ghostingTab.cBox1      = QtWidgets.QCheckBox()
        
        #Inputs
        self.ghostingTab.lineEdit1  = QtWidgets.QLineEdit()
        self.ghostingTab.lineEdit2  = QtWidgets.QLineEdit()
        self.ghostingTab.lineEdit3  = QtWidgets.QLineEdit()
        self.ghostingTab.lineEdit4  = QtWidgets.QLineEdit()
        self.ghostingTab.lineEdit5  = QtWidgets.QLineEdit()
        self.ghostingTab.lineEdit6  = QtWidgets.QLineEdit()
        self.ghostingTab.lineEdit7  = QtWidgets.QLineEdit()
        self.ghostingTab.lineEdit8  = QtWidgets.QLineEdit()
        
        #Labels
        self.ghostingTab.label0 = QtWidgets.QLabel("Exclude ghosting zones")
        self.ghostingTab.label1 = QtWidgets.QLabel("Vessel thresholds, no "     +
                                                       "vessel, small vessel, " +
                                                       "large vessel")
        self.ghostingTab.label2 = QtWidgets.QLabel("Small vessel exclusion zone"+
                                                       " X, Y")
        self.ghostingTab.label3 = QtWidgets.QLabel("Large vessel exclusion zone"+
                                                       " X, Y")
        self.ghostingTab.label4 = QtWidgets.QLabel("Bright pixel percentile")
        
        #Add to layout
        self.ghostingTab.layout     = QtWidgets.QGridLayout()
        self.ghostingTab.layout.addWidget(self.ghostingTab.cBox1,     0,0)
        
        self.ghostingTab.layout.addWidget(QHLine(),                   1,0,1,4)
        
        self.ghostingTab.layout.addWidget(self.ghostingTab.lineEdit1, 2,0)
        self.ghostingTab.layout.addWidget(self.ghostingTab.lineEdit2, 2,1)
        
        self.ghostingTab.layout.addWidget(self.ghostingTab.lineEdit4, 3,0)
        self.ghostingTab.layout.addWidget(self.ghostingTab.lineEdit5, 3,1)
        
        self.ghostingTab.layout.addWidget(self.ghostingTab.lineEdit6, 4,0)
        self.ghostingTab.layout.addWidget(self.ghostingTab.lineEdit7, 4,1)
        
        self.ghostingTab.layout.addWidget(self.ghostingTab.lineEdit8, 5,0)

        self.ghostingTab.layout.addWidget(self.ghostingTab.label0,    0,3)
        self.ghostingTab.layout.addWidget(self.ghostingTab.label1,    2,3)
        self.ghostingTab.layout.addWidget(self.ghostingTab.label2,    3,3)
        self.ghostingTab.layout.addWidget(self.ghostingTab.label3,    4,3)
        self.ghostingTab.layout.addWidget(self.ghostingTab.label4,    5,3)
        
        self.ghostingTab.setLayout(self.ghostingTab.layout)
        
        
    def initResetTab(self):
        self.resetTab.resetButton   = QtWidgets.QPushButton("Reset settings")
        self.resetTab.resetButton.setToolTip(
                "Resets all settings to their default value.")
        self.resetTab.resetButton.pressed.connect(self.reset)
        
        self.resetTab.layout        = QtWidgets.QHBoxLayout()
        self.resetTab.layout.addWidget(self.resetTab.resetButton)
        self.resetTab.setLayout(self.resetTab.layout)
        
        
    def getSettings(self):
        """
        Loads the settings from the QSettings object that is saved with the 
        application and stores them in the UI.
        """
        
        #TODO, don't hardcode this
        COMPANY = "UMCu"
        APPNAME = "SELMA"
        
        settings = QtCore.QSettings(COMPANY, APPNAME)
        
        #General settings
        #==========================================
        
        #medDiam
        medDiam = settings.value("medDiam")
        if medDiam is None:
            medDiam = 53
        self.mainTab.lineEdit1.setText(str(medDiam))
        
        #confidence interval
        confidenceInter = settings.value("confidenceInter")
        if confidenceInter is None:
            confidenceInter = 0.05
        self.mainTab.lineEdit2.setText(str(confidenceInter))
        
        #white matter probability
        whiteMatterProb = settings.value("whiteMatterProb")
        if whiteMatterProb is None:
            whiteMatterProb = 0.5
        self.mainTab.lineEdit3.setText(str(whiteMatterProb))
        
        #average over cardiac cycle
        averageCardiacCycle = settings.value("averageCardiacCycle")
        if averageCardiacCycle is None:
            averageCardiacCycle = True
        else:
            averageCardiacCycle = averageCardiacCycle == 'true'
        self.mainTab.cBox1.setChecked(averageCardiacCycle)
        
        #exclude perpendicular vessels
        excludePerpendicular = settings.value("excludePerpendicular")
        if excludePerpendicular is None:
            excludePerpendicular = True
        else:
            excludePerpendicular = excludePerpendicular == 'true'
        self.mainTab.cBox2.setChecked(excludePerpendicular)
        
        
        #Ghosting settings
        #=============================================
        
        #perform the ghosting filter
        doGhosting = settings.value("doGhosting")
        if doGhosting is None:
            doGhosting = True
        else:
            doGhosting = doGhosting == 'true'
        self.ghostingTab.cBox1.setChecked(doGhosting)
        
        
        #Vessel thresholds
        noVesselThresh = settings.value("noVesselThresh")
        if noVesselThresh is None:
            noVesselThresh = 5
        self.ghostingTab.lineEdit1.setText(str(noVesselThresh))
        
        smallVesselThresh = settings.value("smallVesselThresh")
        if smallVesselThresh is None:
            smallVesselThresh = 20
        self.ghostingTab.lineEdit2.setText(str(smallVesselThresh))
        
        
        #small vessel exclusion zone
        smallVesselExclX = settings.value("smallVesselExclX")
        if smallVesselExclX is None:
            smallVesselExclX = 3
        self.ghostingTab.lineEdit4.setText(str(smallVesselExclX))
        
        smallVesselExclY = settings.value("smallVesselExclY")
        if smallVesselExclY is None:
            smallVesselExclY = 40
        self.ghostingTab.lineEdit5.setText(str(smallVesselExclY))
        
        
        #large vessel exclusion zone
        largeVesselExclX = settings.value("largeVesselExclX")
        if largeVesselExclX is None:
            largeVesselExclX = 5
        self.ghostingTab.lineEdit6.setText(str(largeVesselExclX))
        
        largeVesselExclY = settings.value("largeVesselExclY")
        if largeVesselExclY is None:
            largeVesselExclY = 70
        self.ghostingTab.lineEdit7.setText(str(largeVesselExclY))        
        
        
        #Bright vessel percentile
        brightVesselPerc = settings.value("brightVesselPerc")
        if brightVesselPerc is None:
            brightVesselPerc = 0.5
        self.ghostingTab.lineEdit8.setText(str(brightVesselPerc))    
        
        
    def applySettings(self):
        """First checks if all entered values are correct, then saves all
        values to the QSettings associated with the program."""
        
        COMPANY = "UMCu"
        APPNAME = "SELMA"
        
        settings = QtCore.QSettings(COMPANY, APPNAME)
        
        #General settings
        #=========================================
        
        #median diameter
        
        medDiam = self.mainTab.lineEdit1.text()
        try: 
            medDiam = int(medDiam)
        except:
            self.errorLabel.setText(
                    "Median filter diameter has to be an integer.")
            return
        
        if medDiam %2 == 0:
            self.errorLabel.setText(
                    "Median filter diameter has to be odd")
            return
        # Confidence interval
        
        
        confidenceInter = self.mainTab.lineEdit2.text()
        try: 
            confidenceInter = float(confidenceInter)
        except:
            self.errorLabel.setText(
                    "Confidence interval has to be a number.")
            return
        
        if confidenceInter <= 0 or confidenceInter >=1:
            self.errorLabel.setText(
                    "Confidence interval has to be between 0 and 1.")
            return
        
        
        # White matter probability
        whiteMatterProb = self.mainTab.lineEdit3.text()
        try: 
            whiteMatterProb = float(whiteMatterProb)
        except:
            self.errorLabel.setText(
                    "White matter probabilty has to be a number.")
            return
        
        if whiteMatterProb < 0 or whiteMatterProb > 1:
            self.errorLabel.setText(
                    "White matter probability has to be between 0 and 1.")
            return        
        
        
        # Average over cycle
        averageCardiacCycle = self.mainTab.cBox1.isChecked()
        
        # Exclude perpendicular Vessels
        excludePerpendicular= self.mainTab.cBox2.isChecked()
        
        
        #Ghosting settings
        #=========================================
        
        #perform the ghosting filter
        doGhosting = self.ghostingTab.cBox1.isChecked()
        
        #No Vessel threshold
        noVesselThresh = self.ghostingTab.lineEdit1.text()
        try: 
            noVesselThresh = int(noVesselThresh)
        except:
            self.errorLabel.setText(
                    "No vessel threshold has to be a number.")
            return
        if noVesselThresh < 0:
            self.errorLabel.setText(
                    "No vessel threshold has to be > 0")
            return        
        
        #
        #small Vessel threshold
        smallVesselThresh = self.ghostingTab.lineEdit2.text()
        try: 
            smallVesselThresh = int(smallVesselThresh)
        except:
            self.errorLabel.setText(
                    "Small vessel threshold has to be a number.")
            return
        if smallVesselThresh < 0:
            self.errorLabel.setText(
                    "Small vessel threshold has to be > 0.")
            return        
        
        
        #
        #small vessel exclusion zone - X
        smallVesselExclX = self.ghostingTab.lineEdit4.text()
        try: 
            smallVesselExclX = int(smallVesselExclX)
        except:
            self.errorLabel.setText(
                    "Small vessel exclusion zone X has to be a number.")
            return
        if smallVesselExclX < 0:
            self.errorLabel.setText(
                    "Small vessel exclusion zone X has to be > 0.")
            return        
        
        #
        #small vessel exclusion zone - Y
        smallVesselExclY = self.ghostingTab.lineEdit5.text()
        try: 
            smallVesselExclY = int(smallVesselExclY)
        except:
            self.errorLabel.setText(
                    "Small vessel exclusion zone Y has to be a number.")
            return
        if smallVesselExclY < 0:
            self.errorLabel.setText(
                    "Small vessel exclusion zone Y has to be > 0.")
            return        
        
        #
        #large vessel exclusion zone - X
        largeVesselExclX = self.ghostingTab.lineEdit6.text()
        try: 
            largeVesselExclX = int(largeVesselExclX)
        except:
            self.errorLabel.setText(
                    "Large vessel exclusion zone X has to be a number.")
            return
        if largeVesselExclX < 0:
            self.errorLabel.setText(
                    "Large vessel exclusion zone X has to be > 0.")
            return        
        
        #
        #large vessel exclusion zone - Y
        largeVesselExclY = self.ghostingTab.lineEdit7.text()
        try: 
            largeVesselExclY = int(largeVesselExclY)
        except:
            self.errorLabel.setText(
                    "Large vessel exclusion zone Y has to be a number.")
            return
        if largeVesselExclY < 0:
            self.errorLabel.setText(
                    "Large vessel exclusion zone Y has to be > 0.")
            return        
        
        #
        #Bright vessel percentile
        brightVesselPerc = self.ghostingTab.lineEdit8.text()
        try: 
            brightVesselPerc = float(brightVesselPerc)
        except:
            self.errorLabel.setText(
                    "Bright vessel percentile has to be a number.")
            return
        
        if brightVesselPerc < 0 or brightVesselPerc > 1:
            self.errorLabel.setText(
                    "Bright vessel percentile has to between 0 and 1.")
            return        
        

        #Save all to settings
        #================================
        #================================
        settings.setValue('medDIam',                medDiam)
        settings.setValue('confidenceInter',        confidenceInter)
        settings.setValue('whiteMatterProb',        whiteMatterProb)
        settings.setValue('averageCardiacCycle',    averageCardiacCycle)
        settings.setValue('excludePerpendicular',   excludePerpendicular)
        
        settings.setValue('doGhosting',             doGhosting)
        settings.setValue('noVesselThresh',         noVesselThresh)
        settings.setValue('smallVesselThresh',      smallVesselThresh)
        settings.setValue('smallVesselExclX',       smallVesselExclX)
        settings.setValue('smallVesselExclY',       smallVesselExclY)
        settings.setValue('largeVesselExclX',       largeVesselExclX)
        settings.setValue('largeVesselExclY',       largeVesselExclY)
        settings.setValue('brightVesselPerc',       brightVesselPerc)
        self.close()
        
    def reset(self):
        """Removes all settings from the UI, and saves it to the application,
        prompting the values to reset to their defaults.
        """
        COMPANY = "UMCu"
        APPNAME = "SELMA"
        
        settings = QtCore.QSettings(COMPANY, APPNAME)
        settings.clear()
        settings.sync()
        
        self.getSettings()



# ====================================================================