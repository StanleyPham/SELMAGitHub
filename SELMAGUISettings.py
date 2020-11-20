#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`SelmaSettings`

"""

# ====================================================================

from PyQt5 import (QtCore, QtGui, QtWidgets)
import os

# ====================================================================
class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


def getInfo():
    """Reads the name, companyname and version number from info.txt and 
    returns them."""
    
    fn = "info.txt"
    fullpath = os.path.join(os.getcwd(), fn)
    
    with open (fullpath, "r") as info:
        data=info.readlines()
        company     = data[0]
        appname     = data[1]
        version     = data[2]
        
        return company, appname, version
    
    
        

class SelmaSettings(QtWidgets.QWidget):
    """
    This class contains the UI for interacting with the user settings. 
    Settings are viewed, edited and saved.
    
    The settings window displays the following tabs:
        General         - for the main functioning of the program
        Ghosting        - for the removeGhosting function
        Perpendicular   - for the removeNonPerpendicular function
        Reset           - for resetting the settings to their default
    """
    #Signals
    thresholdSignal     = QtCore.pyqtSignal() 
    
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
        self.nonPerpTab     = QtWidgets.QWidget()
        self.segmentTab     = QtWidgets.QWidget()
        self.resetTab       = QtWidgets.QWidget()
        
        self.tabs.addTab(self.mainTab,      "General")
        self.tabs.addTab(self.ghostingTab,  "Ghosting")
        self.tabs.addTab(self.nonPerpTab,   "Non-Perpendicular")
        self.tabs.addTab(self.segmentTab,   "Segmentation")
        self.tabs.addTab(self.resetTab,     "Reset")
        
        #Design Tabs
        self.initMainTab()
        self.initGhostingTab()
        self.initNonPerpTab()
        self.initSegmentTab()
        self.initResetTab()
        
        #Add to layout
        self.layout         = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.tabs)
        
        #Buttons
        self.okButton       = QtWidgets.QPushButton("OK")
        self.cancelButton   = QtWidgets.QPushButton("Cancel")
        self.applyButton    = QtWidgets.QPushButton("Apply")
        self.okButton.pressed.connect(self.okButtonPushed)
        self.cancelButton.pressed.connect(self.close)
        self.applyButton.pressed.connect(self.applySettings)
        
        self.buttonLayout   = QtWidgets.QHBoxLayout(self)
        self.buttonLayout.addWidget(self.applyButton)
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
        
        self.mainTab.medDiamEdit                = QtWidgets.QLineEdit()
        self.mainTab.confidenceInterEdit        = QtWidgets.QLineEdit()
#        self.mainTab.whiteMatterProbEdit        = QtWidgets.QLineEdit()
        self.mainTab.averageCardiacCycleBox     = QtWidgets.QCheckBox()
        self.mainTab.gaussianSmoothingBox       = QtWidgets.QCheckBox()
        self.mainTab.ignoreOuterBandBox         = QtWidgets.QCheckBox()
        self.mainTab.decimalCommaBox            = QtWidgets.QCheckBox()
        self.mainTab.mmPixelBox                 = QtWidgets.QCheckBox()
        
        self.mainTab.label1     = QtWidgets.QLabel("Median filter diameter")
        self.mainTab.label2     = QtWidgets.QLabel("Confindence interval")
        self.mainTab.label3     = QtWidgets.QLabel("mm")
        self.mainTab.label4     = QtWidgets.QLabel("Average over cardiac cycle")
        self.mainTab.label5     = QtWidgets.QLabel(
            "Use Gaussian smoothing\ninstead of median filter")
        self.mainTab.label6     = QtWidgets.QLabel(
            "Ignore the outer 80 pixels\nof the image.")
        self.mainTab.label7     = QtWidgets.QLabel(
            "Use a decimal comma in the\noutput instead of a dot.")
        
        self.mainTab.label1.setToolTip(
            "Diameter of the kernel used in the median filtering operations.")
        self.mainTab.label2.setToolTip(
            "Confidence interval used to determine whether a vessel is" + 
            " significant. \nDefault is 0.05.")
        self.mainTab.label3.setToolTip(
            "Select whether the diameter is in mm. If off, diameter is in" +
            " pixels. \n If in mm, diameter gets rounded up to the nearest" +
            "odd pixel value.")
        self.mainTab.label4.setToolTip(
            "When toggled on, the reported values per voxel are averaged" +
            " over the cycle.\n If not on, the values are from the first phase.")
        self.mainTab.label5.setToolTip(
            "Speeds up analysis drastically, might yield inaccurate results."+
            "\nUse only for testing.")
        self.mainTab.label6.setToolTip(
            "Removes the outer 80 pixels at each edge from the mask. ")

        #Add items to layout
        self.mainTab.layout     = QtWidgets.QGridLayout()
        self.mainTab.layout.addWidget(self.mainTab.medDiamEdit, 0,0)
        self.mainTab.layout.addWidget(self.mainTab.mmPixelBox, 0,2)
        self.mainTab.layout.addWidget(self.mainTab.confidenceInterEdit, 1,0)
        
        self.mainTab.layout.addWidget(QHLine(),               3,0,1,2)
        
        self.mainTab.layout.addWidget(self.mainTab.averageCardiacCycleBox,
                                      4,0)
        self.mainTab.layout.addWidget(self.mainTab.gaussianSmoothingBox,
                                      5,0)
        self.mainTab.layout.addWidget(self.mainTab.ignoreOuterBandBox,
                                      6,0)
        self.mainTab.layout.addWidget(self.mainTab.decimalCommaBox,
                                      7,0)
        
        #Add labels to layout
        self.mainTab.layout.addWidget(self.mainTab.label1,      0,1)
        self.mainTab.layout.addWidget(self.mainTab.label2,      1,3)
        self.mainTab.layout.addWidget(self.mainTab.label3,      0,3)
        self.mainTab.layout.addWidget(self.mainTab.label4,      4,3)
        self.mainTab.layout.addWidget(self.mainTab.label5,      5,3)
        self.mainTab.layout.addWidget(self.mainTab.label6,      6,3)
        self.mainTab.layout.addWidget(self.mainTab.label7,      7,3)
        
        self.mainTab.setLayout(self.mainTab.layout)
        
    def initGhostingTab(self): 
        """The tab containing the removeGhosting settings."""

        #Toggle Ghosting
        self.ghostingTab.doGhostingBox      = QtWidgets.QCheckBox()
        
        #Inputs
        self.ghostingTab.noVesselThreshEdit     = QtWidgets.QLineEdit()
        self.ghostingTab.smallVesselThreshEdit  = QtWidgets.QLineEdit()
        self.ghostingTab.smallVesselExclXEdit   = QtWidgets.QLineEdit()
        self.ghostingTab.smallVesselExclYEdit   = QtWidgets.QLineEdit()
        self.ghostingTab.largeVesselExclXEdit   = QtWidgets.QLineEdit()
        self.ghostingTab.largeVesselExclYEdit   = QtWidgets.QLineEdit()
        self.ghostingTab.brightVesselPercEdit   = QtWidgets.QLineEdit()
        
        #Labels
        self.ghostingTab.label0 = QtWidgets.QLabel("Exclude ghosting zones")
        self.ghostingTab.label1 = QtWidgets.QLabel("Vessel thresholds"     +
                                                       "small vessel & large vessel")
        self.ghostingTab.label2 = QtWidgets.QLabel("Small vessel exclusion zone"+
                                                       " X, Y")
        self.ghostingTab.label3 = QtWidgets.QLabel("Large vessel exclusion zone"+
                                                       " X, Y")
        self.ghostingTab.label4 = QtWidgets.QLabel("Bright pixel percentile")
        
        #Tooltips
        self.ghostingTab.label0.setToolTip("When toggled on, the areas near particulary"   +
                                       " bright vessels are excluded from analysis.")
        self.ghostingTab.label1.setToolTip("Thresholds for qualifying as a small or large vessel. "             +
                                       "\nAnything smaller than a small vessel is considered not to be a vessel.")
        self.ghostingTab.label2.setToolTip("The ghosting zone around a small vessel is increased in the X- and Y-directions \nby this much.")
        self.ghostingTab.label3.setToolTip("The ghosting zone around a large vessel is increased in the X- and Y-directions \nby this much.")
        self.ghostingTab.label4.setToolTip("What percentage intensity a voxel needs to have in order to be classified as a bright vessel.")
        
        
        #Add to layout
        self.ghostingTab.layout     = QtWidgets.QGridLayout()
        self.ghostingTab.layout.addWidget(self.ghostingTab.doGhostingBox,     0,0)
        
        self.ghostingTab.layout.addWidget(QHLine(),                   1,0,1,4)
        
        self.ghostingTab.layout.addWidget(self.ghostingTab.noVesselThreshEdit, 2,0)
        self.ghostingTab.layout.addWidget(self.ghostingTab.smallVesselThreshEdit, 2,1)
        
        self.ghostingTab.layout.addWidget(self.ghostingTab.smallVesselExclXEdit, 3,0)
        self.ghostingTab.layout.addWidget(self.ghostingTab.smallVesselExclYEdit, 3,1)
        
        self.ghostingTab.layout.addWidget(self.ghostingTab.largeVesselExclXEdit, 4,0)
        self.ghostingTab.layout.addWidget(self.ghostingTab.largeVesselExclYEdit, 4,1)
        
        self.ghostingTab.layout.addWidget(self.ghostingTab.brightVesselPercEdit, 5,0)

        self.ghostingTab.layout.addWidget(self.ghostingTab.label0,    0,3)
        self.ghostingTab.layout.addWidget(self.ghostingTab.label1,    2,3)
        self.ghostingTab.layout.addWidget(self.ghostingTab.label2,    3,3)
        self.ghostingTab.layout.addWidget(self.ghostingTab.label3,    4,3)
        self.ghostingTab.layout.addWidget(self.ghostingTab.label4,    5,3)
        
        self.ghostingTab.setLayout(self.ghostingTab.layout)
        
    def initNonPerpTab(self):
        """The tab containing the removeGhosting settings."""

        #Toggle nonPerp
        self.nonPerpTab.removeNonPerpBox       = QtWidgets.QCheckBox()
        
        #Inputs
        self.nonPerpTab.removePerpXEdit         = QtWidgets.QLineEdit()
        self.nonPerpTab.removePerpYEdit         = QtWidgets.QLineEdit()
        self.nonPerpTab.removePerpMagThreshEdit = QtWidgets.QLineEdit()
        self.nonPerpTab.removePerpRatioThreshEdit  = QtWidgets.QLineEdit()
        
        #Labels
        self.nonPerpTab.label0 = QtWidgets.QLabel("Exclude non-perpendicular zones")
        self.nonPerpTab.label1 = QtWidgets.QLabel("Window size for measuring vessel shape.")
        self.nonPerpTab.label2 = QtWidgets.QLabel("Magnitude threshold for measuring vessel shape.")
        self.nonPerpTab.label3 = QtWidgets.QLabel("Major / minor axis threshold ratio.")
        #Tooltips        
        self.nonPerpTab.label0.setToolTip("When toggled on, the significant vessels"   +
                                       " are filtered for any vessels that lie \n"  +
                                       "perpendicular to the imaging plane.")
        self.nonPerpTab.label1.setToolTip("The vessel shape is measured in a window\naround each cluster. These are\nthe window dimensions (X,Y).")
        self.nonPerpTab.label2.setToolTip("The vessel shape is measured on the\nmagnitude data. This is the magnitude\nthreshold for vessels.")
        self.nonPerpTab.label3.setToolTip("The criterion for exlusion is: major\nradius / minor radius > X.\n This is X")
        
        #Add to layout
        self.nonPerpTab.layout              = QtWidgets.QGridLayout()
        self.nonPerpTab.layout.addWidget(self.nonPerpTab.removeNonPerpBox,     0,0)
        
        self.nonPerpTab.layout.addWidget(QHLine(),                   1,0,1,4)
        
        self.nonPerpTab.layout.addWidget(self.nonPerpTab.removePerpXEdit, 2,0)
        self.nonPerpTab.layout.addWidget(self.nonPerpTab.removePerpYEdit, 2,1)
        
        self.nonPerpTab.layout.addWidget(self.nonPerpTab.removePerpMagThreshEdit, 3,0)
        self.nonPerpTab.layout.addWidget(self.nonPerpTab.removePerpRatioThreshEdit,  4,0)
        
        self.nonPerpTab.layout.addWidget(self.nonPerpTab.label0,    0,3)
        self.nonPerpTab.layout.addWidget(self.nonPerpTab.label1,    2,3)
        self.nonPerpTab.layout.addWidget(self.nonPerpTab.label2,    3,3)
        self.nonPerpTab.layout.addWidget(self.nonPerpTab.label3,    4,3)
        
        self.nonPerpTab.setLayout(self.nonPerpTab.layout)
        
    def initSegmentTab(self):
        
        self.segmentTab.whiteMatterProb            = QtWidgets.QLineEdit()
        
        self.segmentTab.label1     = QtWidgets.QLabel("White matter probability")

        #Add items to layout
        self.segmentTab.layout     = QtWidgets.QGridLayout()
        self.segmentTab.layout.addWidget(self.segmentTab.whiteMatterProb, 0,0)
        
        #Add labels to layout
        self.segmentTab.layout.addWidget(self.segmentTab.label1,      0,1)
        
        self.segmentTab.setLayout(self.segmentTab.layout)
        
        #Add button to layout
        self.segmentTab.applyButton     = QtWidgets.QPushButton("Apply")
        
        
        
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
        medDiam                 = settings.value("medDiam")
        if medDiam is None:
            medDiam             = 10
        self.mainTab.medDiamEdit.setText(str(medDiam))
        
        mmPixel                 = settings.value("mmPixel")
        if mmPixel is None:
            mmPixel     = True
        else:
            mmPixel = mmPixel == 'true'
        self.mainTab.mmPixelBox.setChecked(mmPixel)
        
        #confidence interval
        confidenceInter         = settings.value("confidenceInter")
        if confidenceInter is None:
            confidenceInter     = 0.05
        self.mainTab.confidenceInterEdit.setText(str(confidenceInter))
        
        #white matter probability
#        whiteMatterProb         = settings.value("whiteMatterProb")
#        if whiteMatterProb is None:
#            whiteMatterProb     = 0.5
#        self.mainTab.whiteMatterProbEdit.setText(str(whiteMatterProb))
        
        #average over cardiac cycle
        averageCardiacCycle     = settings.value("averageCardiacCycle")
        if averageCardiacCycle is None:
            averageCardiacCycle = True
        else:
            averageCardiacCycle = averageCardiacCycle == 'true'
        self.mainTab.averageCardiacCycleBox.setChecked(averageCardiacCycle)
        
        #Do Gaussian smoothing - default is False
        gaussianSmoothing       = settings.value("gaussianSmoothing")
        if gaussianSmoothing is None:
            gaussianSmoothing   = False
        else:
            gaussianSmoothing   = gaussianSmoothing == 'true'
        self.mainTab.gaussianSmoothingBox.setChecked(gaussianSmoothing)
        
        #Ignore outer band
        ignoreOuterBand         = settings.value("ignoreOuterBand")
        if ignoreOuterBand is None:
            ignoreOuterBand = True
        else:
            ignoreOuterBand     = ignoreOuterBand == 'true'
        self.mainTab.ignoreOuterBandBox.setChecked(ignoreOuterBand)
        
        #Use decimal comma
        decimalComma         = settings.value("decimalComma")
        if decimalComma is None:
            decimalComma = True
        else:
            decimalComma     = decimalComma == 'true'
        self.mainTab.decimalCommaBox.setChecked(decimalComma)
        
        #Ghosting settings
        #=============================================
        
        #perform the ghosting filter
        doGhosting = settings.value("doGhosting")
        if doGhosting is None:
            doGhosting = True
        else:
            doGhosting = doGhosting == 'true'
        self.ghostingTab.doGhostingBox.setChecked(doGhosting)
        
        
        #Vessel thresholds
        noVesselThresh = settings.value("noVesselThresh")
        if noVesselThresh is None:
            noVesselThresh = 5
        self.ghostingTab.noVesselThreshEdit.setText(str(noVesselThresh))
        
        smallVesselThresh = settings.value("smallVesselThresh")
        if smallVesselThresh is None:
            smallVesselThresh = 20
        self.ghostingTab.smallVesselThreshEdit.setText(str(smallVesselThresh))
        
        
        #small vessel exclusion zone
        smallVesselExclX = settings.value("smallVesselExclX")
        if smallVesselExclX is None:
            smallVesselExclX = 3
        self.ghostingTab.smallVesselExclXEdit.setText(str(smallVesselExclX))
        
        smallVesselExclY = settings.value("smallVesselExclY")
        if smallVesselExclY is None:
            smallVesselExclY = 40
        self.ghostingTab.smallVesselExclYEdit.setText(str(smallVesselExclY))
        
        
        #large vessel exclusion zone
        largeVesselExclX = settings.value("largeVesselExclX")
        if largeVesselExclX is None:
            largeVesselExclX = 5
        self.ghostingTab.largeVesselExclXEdit.setText(str(largeVesselExclX))
        
        largeVesselExclY = settings.value("largeVesselExclY")
        if largeVesselExclY is None:
            largeVesselExclY = 70
        self.ghostingTab.largeVesselExclYEdit.setText(str(largeVesselExclY))        
        
        
        #Bright vessel percentile
        brightVesselPerc = settings.value("brightVesselPerc")
        if brightVesselPerc is None:
            brightVesselPerc = 0.997
        self.ghostingTab.brightVesselPercEdit.setText(str(brightVesselPerc)) 
        
        
        
        #Non Perpendicular settings
        #=============================================
        
        #RemoveNonPerpendicular
        removeNonPerp = settings.value("removeNonPerp")
        if removeNonPerp is None:
            removeNonPerp = True
        else:
            removeNonPerp = removeNonPerp == 'true'
        self.nonPerpTab.removeNonPerpBox.setChecked(removeNonPerp)
        
        
        #Window
        removePerpX = settings.value("removePerpX")
        if removePerpX is None:
            removePerpX = 5
        self.nonPerpTab.removePerpXEdit.setText(str(removePerpX))
        
        removePerpY = settings.value("removePerpY")
        if removePerpY is None:
            removePerpY = 5
        self.nonPerpTab.removePerpYEdit.setText(str(removePerpY))
        
        
        #magnitude threshold
        removePerpMagThresh = settings.value("removePerpMagThresh")
        if removePerpMagThresh is None:
            removePerpMagThresh = 0.8
        self.nonPerpTab.removePerpMagThreshEdit.setText(
            str(removePerpMagThresh))
        
        
        #Exclusion ratio threshold
        removePerpRatioThresh = settings.value("removePerpRatioThresh")
        if removePerpRatioThresh is None:
            removePerpRatioThresh = 5
        self.nonPerpTab.removePerpRatioThreshEdit.setText(
            str(removePerpRatioThresh))
        
        
        #Segmentation settings
        #=============================================
        
        #Brain tissue probability
        whiteMatterProb = settings.value("whiteMatterProb")
        if whiteMatterProb is None:
            whiteMatterProb = 0.5
        self.segmentTab.whiteMatterProb.setText(str(whiteMatterProb))
        
        
        
    def okButtonPushed(self):
        self.applySettings()
        self.close()        
        
    def applySettings(self):
        """First checks if all entered values are correct, then saves all
        values to the QSettings associated with the program."""
        
        COMPANY = "UMCu"
        APPNAME = "SELMA"
        
        settings = QtCore.QSettings(COMPANY, APPNAME)
        
        #General settings
        #=========================================
        
        #median diameter
        
        medDiam = self.mainTab.medDiamEdit.text()
        try: 
            medDiam = int(medDiam)
        except:
            self.errorLabel.setText(
                    "Median filter diameter has to be an integer.")
            return
        
        if medDiam %2 == 0 and not self.mainTab.mmPixelBox.isChecked():
            self.errorLabel.setText(
                "Median filter diameter has to be an odd number of pixels.")
            return
        
        
        #Pixel or mm diameter:
        mmPixel     = self.mainTab.mmPixelBox.isChecked()      
        
        
        # Confidence interval
        
        confidenceInter = self.mainTab.confidenceInterEdit.text()
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
#        whiteMatterProb = self.mainTab.whiteMatterProbEdit.text()
#        try: 
#            whiteMatterProb = float(whiteMatterProb)
#        except:
#            self.errorLabel.setText(
#                    "White matter probabilty has to be a number.")
#            return
#        
#        if whiteMatterProb < 0 or whiteMatterProb > 1:
#            self.errorLabel.setText(
#                    "White matter probability has to be between 0 and 1.")
#            return        
        
        
        # Average over cycle
        averageCardiacCycle = self.mainTab.averageCardiacCycleBox.isChecked()
        gaussianSmoothing   = self.mainTab.gaussianSmoothingBox.isChecked()
        ignoreOuterBand     = self.mainTab.ignoreOuterBandBox.isChecked()
        decimalComma        = self.mainTab.decimalCommaBox.isChecked()
        
        
        #=========================================
        #=========================================
        #           Ghosting settings
        #=========================================
        #=========================================
        #perform the ghosting filter
        doGhosting = self.ghostingTab.doGhostingBox.isChecked()
        
        #No Vessel threshold
        noVesselThresh = self.ghostingTab.noVesselThreshEdit.text()
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
        smallVesselThresh = self.ghostingTab.smallVesselThreshEdit.text()
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
        smallVesselExclX = self.ghostingTab.smallVesselExclXEdit.text()
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
        smallVesselExclY = self.ghostingTab.smallVesselExclYEdit.text()
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
        largeVesselExclX = self.ghostingTab.largeVesselExclXEdit.text()
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
        largeVesselExclY = self.ghostingTab.largeVesselExclYEdit.text()
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
        brightVesselPerc = self.ghostingTab.brightVesselPercEdit.text()
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
        

        #=========================================
        #=========================================
        #           Non-Perpendicular
        #=========================================
        #=========================================

        #RemoveNonPerpendicular
        removeNonPerp = self.nonPerpTab.removeNonPerpBox.isChecked()        
        
        #
        #Window
        #X
        removePerpX = self.nonPerpTab.removePerpXEdit.text()
        try: 
            removePerpX = float(removePerpX)
        except:
            self.errorLabel.setText(
                    "Non-perpendicular window selection X has to be a number.")
            return
        
        if removePerpX < 0:
            self.errorLabel.setText(
                    "Non-perpendicular window selection X has to >0.")
            return     
        
        #Y
        
        removePerpY = self.nonPerpTab.removePerpYEdit.text()
        try: 
            removePerpY = float(removePerpY)
        except:
            self.errorLabel.setText(
                    "Non-perpendicular window selection Y has to be a number.")
            return
        
        if removePerpY < 0:
            self.errorLabel.setText(
                    "Non-perpendicular window selection Y has to > 0.")
            return     
        
        
        #
        #Magnitude Percentage Threshold
        removePerpMagThresh = self.nonPerpTab.removePerpMagThreshEdit.text()
        try: 
            removePerpMagThresh = float(removePerpMagThresh)
        except:
            self.errorLabel.setText(
                    "Non-perpendicular window selection X has to be a number.")
            return
        
        if removePerpMagThresh < 0 or removePerpMagThresh > 1:
            self.errorLabel.setText(
                    "Non-perpendicular window selection X has to between 0 and 1.")
            return  
        
        
        #
        #Ratio Threshold
        removePerpRatioThresh = self.nonPerpTab.removePerpRatioThreshEdit.text()
        try: 
            removePerpRatioThresh = float(removePerpRatioThresh)
        except:
            self.errorLabel.setText(
                    "Non-perpendicular window selection X has to be a number.")
            return
        
        if removePerpRatioThresh < 0:
            self.errorLabel.setText(
                    "Non-perpendicular window selection X has to be > 0.")
            return   
        
        
        
        #=========================================
        #=========================================
        #           Segmentation
        #=========================================
        #=========================================       
        
        #
        #Brain mask
        #Prob
        whiteMatterProb = self.segmentTab.whiteMatterProb.text()
        try: 
            whiteMatterProb = float(whiteMatterProb)
        except:
            self.errorLabel.setText(
                    "White matter probability has to be a number.")
            return
        
        if whiteMatterProb < 0 or whiteMatterProb > 1:
            self.errorLabel.setText(
                    "White matter probability has to be between 0 and 1.")
            return     
        
         

        #Save all to settings
        #================================
        #================================
        #Main
        settings.setValue('medDiam',                medDiam)
        settings.setValue('confidenceInter',        confidenceInter)
        settings.setValue('mmPixel',                mmPixel)
#        settings.setValue('whiteMatterProb',        whiteMatterProb)
        settings.setValue('averageCardiacCycle',    averageCardiacCycle)
        settings.setValue('gaussianSmoothing',      gaussianSmoothing)
        settings.setValue('ignoreOuterBand',        ignoreOuterBand)
        settings.setValue('decimalComma',           decimalComma)
        
        #Ghosting
        settings.setValue('doGhosting',             doGhosting)
        settings.setValue('noVesselThresh',         noVesselThresh)
        settings.setValue('smallVesselThresh',      smallVesselThresh)
        settings.setValue('smallVesselExclX',       smallVesselExclX)
        settings.setValue('smallVesselExclY',       smallVesselExclY)
        settings.setValue('largeVesselExclX',       largeVesselExclX)
        settings.setValue('largeVesselExclY',       largeVesselExclY)
        settings.setValue('brightVesselPerc',       brightVesselPerc)
        
        #nonPerp
        settings.setValue('removeNonPerp',          removeNonPerp)
        settings.setValue('removePerpX',            removePerpX)
        settings.setValue('removePerpY',            removePerpY)
        settings.setValue('removePerpMagThresh',    removePerpMagThresh)
        settings.setValue('removePerpRatioThresh',  removePerpRatioThresh)
        
        #Segmentation
        settings.setValue('whiteMatterProb',        whiteMatterProb)
        
        
        
        #Send signals
        self.thresholdSignal.emit()
        
        
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