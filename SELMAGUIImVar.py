#!/usr/bin/env python

"""
This module contains the following classes:

+ :class:`SelmaSettings`

"""

# ====================================================================

from PyQt5 import (QtCore, QtGui, QtWidgets)

# ====================================================================
class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        

class SelmaImVar(QtWidgets.QWidget):
    """
    This class contains the UI for interacting with the image variables
    loaded in the .dcm file.
    
    The settings window displays the following tabs:
        General     - for the main functioning of the program
        Ghosting    - for the removeGhosting function
        Reset       - for resetting the settings to their default
    """
    
    
    def __init__(self, signalObj):
        QtWidgets.QWidget.__init__(self)
        
        self.signalObj  = signalObj
        
        #Create window
        self.initGui()
    
    def initGui(self):
        self.setGeometry(QtCore.QRect(100, 100, 300, 200))
        self.setWindowTitle("Image Variables")
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        
        #Layout of the variables
        self.vencLabel  = QtWidgets.QLabel("Velocity Encoding (cm/s)")
        self.vencEdit   = QtWidgets.QLineEdit()
        
        self.varLayout  = QtWidgets.QGridLayout()
        self.varLayout.addWidget(self.vencLabel,   0,0)
        self.varLayout.addWidget(self.vencEdit,    0,1)
        
        
        #Layout of the buttons
        self.okButton       = QtWidgets.QPushButton("OK")
        self.cancelButton   = QtWidgets.QPushButton("Cancel")
        self.okButton.pressed.connect(self.okButtonPressed)
        self.cancelButton.pressed.connect(self.close)
        
        self.buttonLayout   = QtWidgets.QHBoxLayout()
        self.buttonLayout.addWidget(self.okButton)
        self.buttonLayout.addWidget(self.cancelButton)
        
        #Window layout
        self.errorLabel     = QtWidgets.QLabel("")
        self.errorLabel.setStyleSheet("QLabel {color: red }")
        
        self.layout         = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.varLayout)
        self.layout.addLayout(self.buttonLayout)
        self.layout.addWidget(self.errorLabel)
        
        self.setLayout(self.layout)
    
    def getVariables(self):
        """
        Gets the necessary variables from the program.
        """
        self.signalObj.getVarSignal.emit()
        
    @QtCore.pyqtSlot(dict)
    def listenForVars(self, variables):
        """Extract the variables from the dictionary that was sent and store 
        them in the window."""
        
        venc    = variables["venc"]
        self.vencEdit.setText(str(venc))

    def okButtonPressed(self):
        """
        Collects all the values from the window, stores them in a dictionary 
        and sends them to the SelmaData class to be managed there.
        
        """
        
        #The dictionary
        res             = dict()
        
        #All the variables
        venc            = self.vencEdit.text()
        try:
            venc        = float(venc)
        except:
            self.errorLabel.setText(
                    "Velocity Encoding has to be a number")
            return        
        if venc == 0:
            self.errorLabel.setText(
                    "Velocity Encoding cannot be 0")
            return
        res["venc"]     = venc

        # other variables...
        
        
        #Send variables to DataModels        
        self.signalObj.setVarSignal.emit(res)
        self.close()
        
        
        


# ====================================================================