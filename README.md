# SELMA

** *Trial-ready Small Vessel MRI Markers* **

**SELMA User Guide**


**Table of Contents**
    • Getting Started
        ◦ Installation
        ◦ Loading Dicoms
        ◦ Viewing the Dicom
        ◦ Applying masks
    • Explanation of algorithm
    • Settings
    
**Getting Started**

The SELMA application is used for extracting and analysing flow information in cerebral vessels.


**Installation**

The program can be installed by cloning this repository. The easiest way of getting (and maintaining) the code is via:
https://desktop.github.com/
A short tutorial on how the software works is available at:
https://help.github.com/en/desktop/getting-started-with-github-desktop

The program can then compiled with any python distribution. The recommended method is to use Anaconda:
https://www.anaconda.com/

In order to run the program, certain packages are needed. These need to be installed in the correct environment.
The easiest way to do this is to create a new Anaconda environment from the selmaEnvironment.yml file.

To do this, launch the 'Anaconda Prompt (Anaconda3)' program, navigate to the directory where the selmaEnvironment.yml file is located, and type the following command:
`conda env create -f environment.yml`

Alternatively, the correct packages can be found by trying to run the program and whenever an error pops up regarding a missing module, installing that module via anaconda or pip. This is not recommended.


**Loading Dicoms**

A dicom can be loaded via the ‘file’-menu in the menubar, or by pressing Ctrl+O. The Dicom must contain magnitude and velocity in order for the analysis to be performed. These can be in any order, but it is assumed that this order is the same for both types, such that each n-th velocity frame can be matched to the n-th magnitude frame.
The program currently only supports Dicoms acquired on Philips machines. Classic Dicom file are also not supported.

**Viewing the Dicom**

After opening a Dicom file, the first of its frames is displayed. The frames can be cycled through by scrolling the mousewheel. The mousewheel can also be used to zoom in and out on the image on the screen if the Control key is pressed. 
Moving and rescaling the image is also possible with the Scroll- and Zoom-menus in the menubar. 
Lastly, the brightness and contrast of the image can be changed by moving the mouse while pressing the middle mouse button. Moving the mouse in the vertical direction changes the brightness, while the horizontal direction changes the contrast. 

**Applying Masks**
The application only reports the analysis of vessels that are contained in a mask. When no mask is supplied, the program will not give any output. Masks can be applied in three different ways:
*  Segmenting -  A mask can be segmented from a T1 dicom by selecting the Segment Mask option from the mask menu in the menubar. This takes a while to do. **This is currently not implemented fully**
*  Loading from files - A pregenerated mask can be applied to the image by selecting the ‘Load Mask’ option from the mask menu in the menubar. The program currently supports .png, .npy, and .mat files.
*  Drawing - A mask can be drawn on top of the currently displayed frame. This is done by pressing the left mouse button and moving the mouse. An exclusion zone can also be drawn when the right mouse button is pressed. 

The latter can be combined with the earlier options: after segmenting or loading a mask, it can be added to / removed from by drawing additional inclusion / exclusion zones.
When many changes are made, or after segmentation, it is recommended to save the mask to make sure that the process doesn’t need to be repeated in the future. This is done with the ‘Save Mask’ option in the mask menu in the menubar. 

**Explanation of Algorithm**

The next step is the analysis...


**Settings**
The behavior can be changed by...