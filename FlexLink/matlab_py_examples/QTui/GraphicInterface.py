# File:       P25_Tester.py
# Author:     Andreas Schwarzinger (8USNT - Rohde & Schwarz USA)
# Notes:      This application is used to test and simulate many of the P25 processing tasks
#             that need to be ported into the C++ P25Scanner.dll

# Some short cuts
# Use Ctr+K Ctr+C to   comment out multiple lines of code (Visual Studio 2017)
# Use Ctr+K Ctr+U to uncomment out multiple lines of code (Visual Studio 2017)
# Use Ctr+K O     to create a new code window             (Visual Studio Code)

__title__     = "P25Simulator"
__author__    = "Andreas Schwarzinger <8USNT - Rohde & Schwarz USA>"
__status__    = "preliminary"
__version__   = "0.1.0.0"
__date__      = "March, 21, 2024"
__copyright__ = 'Copyright 2024 by Rohde & Schwarz USA'


# Path extensions
import sys         # We use sys to append the path (see below and to exit(0) the application)
import ctypes      # Needed for DPI resolution when dealing with multiple monitors (see starting code at the bottom of this file)
import numpy as np

# Import of QT modules
from PyQt5              import QtCore, QtGui, QtWidgets
from PyQt5.QtGui        import QPixmap
from PyQt5.QtWidgets    import QApplication, QMainWindow, QFileDialog, QMessageBox   # QComboBox, QLabel, QWidget, QFileDialog


import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # Needed to display matplotlib figures
from matplotlib.figure                  import Figure             # Needed to display matplotlib figures

import Visualization
vis = Visualization.Visualization()

# This nice feature will display matlibplots with a dark background and white text. Very nice looking.
plt.style.use('dark_background')

# Get the graphics interface created in QtDesigner
from LeonardUI       import Ui_MainWindow  # The GUI interface
from ofdm_802ii_call import ofdm_802ii_call

# import sys
import os

# this assumes that you are running from a path located where the
# code is also located. 
# Specific directory name you want to change to
working_directory = "QTui"

# Get the full path of the script
script_path = os.path.abspath(sys.argv[0])

# Split the path into directories
path_parts = script_path.split(os.sep)

# Check if the specific directory name is in the path and reconstruct the path up to that directory
if working_directory in path_parts:
    # Find the index of the specific directory
    index = path_parts.index(working_directory)
    
    # Reconstruct the path up to the specific directory
    new_path = os.sep.join(path_parts[:index + 1])

    # Change the current working directory
    os.chdir(new_path)
    
    print(f"Changed current working directory to '{new_path}'")
    sys.path.insert(1, '.\\KoliberEng')
    print("inserting KoliberEng to sys.path ")

else:
    print(f"The directory '{working_directory}' does not exist in the path")
    exit



# ----------------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
# Main window class that inherits from QtWidgets.QMainWindow and we need to call the constructor for that base class.           #
# Now create a member variable called ui that will be of type Ui_MainWindow and then call its setupUi function.                 #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
# ----------------------------------------------------------------------------------------------------------------------------- #   
class CMyGui(QMainWindow):

    # -----------------------------------------------------------------------
    # Function: __init__
    # -----------------------------------------------------------------------
    def __init__(self):
        # Call the base class constructor
        super(CMyGui, self).__init__()

        # --------------------------------------------------------------------------------
        # > Create an Ui_MainWindow object and call its setupUi function
        # --------------------------------------------------------------------------------
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # --------------------------------------------------------------------------------
        # > Member variables
        # --------------------------------------------------------------------------------
        self.TxParamsDict             = {}
        
        # --------------------------------------------------------------------------------
        # > Create a canvas that you can draw on later for the Matplotlib drawings
        # --------------------------------------------------------------------------------
        self.sc0 = MplCanvas(self, version = 0, width=8, height = 5, dpi = 100)   

        # Create a vertical box layout and add the self.sc0 canvas widget to it
        self.layout0 = QtWidgets.QVBoxLayout()

        # Add a layout to each widget
        self.layout0.addWidget(self.sc0)

        # Add the layout to the GraphWidget. Now you can plot to your heart content
        self.ui.WiMatPlotLib.setLayout(self.layout0)

        # -------------------------------------------------------------------------------- #
        #                                 Button Controls                                  # 
        # -------------------------------------------------------------------------------- #
        # -----------------------------------
        # The PushButtonApply Button
        self.ui.PushButtonApply.setStyleSheet("QPushButton         {background-color : #505050; color: rgb(255, 221, 187); font: 10pt 'Consolas'; border: 2px solid grey}"
                                              "QPushButton::pressed{background-color : #AAAAAA; color: rgb(0, 0, 0);}}"
                                              "QToolTip            {color:black}}")
        self.ui.PushButtonApply.setToolTip("Read in parameters and run Simulation.")
        self.ui.PushButtonApply.setToolTipDuration(10000)
        self.ui.PushButtonApply.clicked.connect(lambda:    self.ButtonHandler(0))


        # ------------------------------------------------------------------------------------------------
        # > Create callbacks for the Menu items
        # ------------------------------------------------------------------------------------------------
        self.ui.menuFile.setStyleSheet("QMenu::pressed{background-color : #AAAAAA; color : #222222}")
        self.ui.actionExit.triggered.connect(lambda: sys.exit())


        # --------------------------------------------------------------------------------------- #
        # -                                 Text Edit Controls                                 -  #
        # --------------------------------------------------------------------------------------- #
        self.ui.TextEditParameters.setStyleSheet("QTextEdit         {font: 11pt 'Consolas';}"
                                                 "QToolTip          {color:black}}")
        self.ui.TextEditParameters.setToolTip("Declare parameters of the Simulation.")
        
        InitialText = 'd_Carr_Freq_off:  0\nd_IQ_amp_off:     0\nd_IQ_Ph_off:      0\nd_samp_rate_off:  0\nd_Fr_time_off:    0'
        self.ui.TextEditParameters.setText(InitialText);
 
         


    # --------------------------------------------------------------------------------------------------- # 
    #                                                                                                     #
    #                                                                                                     # 
    # Function: ButtonHandler()                                                                           #
    #                                                                                                     #
    #                                                                                                     #
    # --------------------------------------------------------------------------------------------------- #
    def ButtonHandler(self, ButtonIndex: int):
        """
        brief: This function is the generic handler for many buttons in the GUI.
        """
        if(ButtonIndex == 0):
            ParameterString = self.ui.TextEditParameters.toPlainText()
            self.FetchTxParameters(ParameterString)
        
        # --------------------------------------
        # Plot the power spectrum
        # --------------------------------------
        carrFrqOff  = self.TxParamsDict['Carrier Freq off']
        iqAmpOff  = self.TxParamsDict['IQ Gain Imbalance']
        iqPhOff  = self.TxParamsDict['IQ Phase'] 
        sampTimeOff = self.TxParamsDict['Sample Rate Off'] 
        frTimeOff  = self.TxParamsDict['Frame time Off']
        
        d_frq   = (carrFrqOff  / 100.00) # carrier frequency offset
        d_tm    = (frTimeOff  / 100.00)  # Frame time offset range from -1 to 1
        d_clk   = (sampTimeOff / 100.00) #d_Fs, sampling time offset (Fs) range from -1 to 1
        d_iqgn  = (iqAmpOff * 2 / 1000.0) # iq gain imbalance range from -0.2 to 0.2
        d_iqphs = (iqPhOff * 2 / 1000.0) # iq phase imbalance  range from -0.2 to 0.2
        d_iqdt  = 0.0 # iq differential timeme  range from -0.2 to 0.2
        

        # function definition: def ofdm_802ii_call(d_frq, d_tm, d_clk, d_gn, d_phs, del_t):
        fdet2 = ofdm_802ii_call(d_frq, d_tm, d_clk, d_iqgn, d_iqphs, d_iqdt)
        # look for initial text to see the labels on the ui 
        # currently:         
        # InitialText = 'd_Carr_Freq_off:  0\nd_IQ_amp_off:   0\nd_IQ_Ph_off:   0\nd_samp_rate_off (Hz): 0\nd_Fr_time_off: 0'


        self.plot_spectrum(fdet2=fdet2)
        vis.plot_constellation(fdet2,name='IQ constellation plot')

        # n  = np.arange(0, N, 1, np.int32)
        # ComplexSinusoid = A * np.exp(1j*(np.pi*(n/Fs)*F + P))

        # self.sc0.axes.cla()
        # self.sc0.axes.plot(n, ComplexSinusoid.real, 'r', n, ComplexSinusoid.imag, 'b')
        # self.sc0.axes.set_title('Plot of a complex sinusoid')
        # self.sc0.axes.set_xlabel('n')
        # self.sc0.axes.grid(color ='#333333')
        # self.sc0.draw()

    def plot_spectrum(self, fdet2):
        # Visualization, now ensuring the x and y arrays are correctly matched
        x_plot = np.linspace(-0.5, 0.5, num=fdet2.shape[1])  # Ensure x_plot matches the second dimension of fdet2
        # x_plot = np.arange(-0.5, 0.5, 1/56)
        
        self.sc0.axes.cla()
        
        for mm in range(fdet2.shape[0]):
            self.sc0.axes.plot(x_plot, np.real(fdet2[mm,:]), 'ro')
            
        self.sc0.axes.set_title('Real Spectrum (In-Phase)')        
        self.sc0.axes.set_xlabel('Normalized Frequency')
        self.sc0.axes.set_ylabel('Amplitude')
        self.sc0.axes.grid(color ='#333333')
        self.sc0.draw()
        
        # for figure_num, title in zip([3, 4, 5], ['Real Spectrum (In-Phase)', 'Imaginary Spectrum (Quad-Phase)', 'Constellation Diagram']):
        #     plt.figure(figure_num)
        #     plt.clf()
        #     if figure_num != 5:
        #         plt.plot(x_plot, np.real(fdet2).T, 'or')  # Corrected plotting call
        #     else:
        #         plt.scatter(np.real(fdet2).flatten(), np.imag(fdet2).flatten(), color='red')
        #     plt.grid(True)
        #     plt.title(title)
        #     plt.xlabel('Normalized Frequency' if figure_num != 5 else '')
        #     plt.ylabel('Amplitude' if figure_num != 5 else '')
        #     plt.axis('square' if figure_num == 5 else None)
        #     plt.show()

    # --------------------------------------------------------------------------------------------------- # 
    #                                                                                                     #
    #                                                                                                     # 
    # Miscellaneous support functions                                                                     #
    #                                                                                                     #
    #                                                                                                     #
    # --------------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------------------
    # This member function will remove spaces from text that was split
    # ---------------------------------------------------------------------------------------------------------
    def StripText(  self
                  , Text: str):
        '''
        This function will strip all text from the text string 'Text', leaving only the numbers in StringList
        '''
        # --------------------------------
        # Error checking
        assert isinstance(Text, str)

        # Replace all occurrence of "\n", by " "
        Text1 = Text.replace("\n", " ")

        # Split string using seperate " "
        StringList = Text1.split(" ")
            
        # Replace all strings that are not numbers by ''
        for Index, String in enumerate(StringList):
            try:
                float(String)   # if this fails, then it was not a number
            except:
                StringList[Index] = ''
        
        # Remove all entries in the list that are equal to ''
        while(True):
            try:
                StringList.remove('')
            except:
                break
            
        return StringList



    # ---------------------------------------------------------------------------------------------------------
    # This function will fetch the channel parameters in TeTxParameters.
    # ---------------------------------------------------------------------------------------------------------
    def FetchTxParameters(self, Text: str) -> dict:
        '''
        This function will fetch the channel parameters in TeTxParameters.
        '''
        StringList         = self.StripText(Text)
        NumberOfParameters = len(StringList)
        assert  NumberOfParameters == 5,          "Detected the wrong number of parameters."
        
        self.TxParamsDict = {}
        # self.TxParamsDict['Frequency']    = float(StringList[0])
        # self.TxParamsDict['Amplitude']    = float(StringList[1])
        # self.TxParamsDict['Phase']        = float(StringList[2]) 
        # self.TxParamsDict['SampleRate']   = float(StringList[3]) 
        # self.TxParamsDict['NumSamples']   = int(StringList[4]) 
        self.TxParamsDict['Carrier Freq off']   = float(StringList[0])
        self.TxParamsDict['IQ Gain Imbalance']  = float(StringList[1])
        self.TxParamsDict['IQ Phase']           = float(StringList[2]) 
        self.TxParamsDict['Sample Rate Off']    = float(StringList[3]) 
        self.TxParamsDict['Frame time Off']     = float(StringList[4]) 

        return self.TxParamsDict


# --------------------------------------------------------------------------------------------
# We will use the MplCanvas class, which inherits from FigureCanvasQTAgg to draw MatPlotLib figures
class MplCanvas(FigureCanvasQTAgg):
    # ---------------------------------------------
    # Function: __init__()
    # ---------------------------------------------
    def __init__(self
               , parent  = None
               , version = 0        # Version 0 - subplot(111)  # Version 1 - subplot(121) and subplot(122)
               , width   = 2
               , height  = 3
               , dpi     = 100):

        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        #fig.patch.set_facecolor("white")
        
        if version == 0:
            self.axes    = fig.add_subplot(111)
        elif version == 1:
            self.axes121 = fig.add_subplot(121)
            self.axes122 = fig.add_subplot(122)
        else:
            self.axes211 = fig.add_subplot(211)
            self.axes212 = fig.add_subplot(212)
        super().__init__(fig)






# ---------------------------------------------------------------
#
#
# The start Code
#
#
# ---------------------------------------------------------------
if __name__ == "__main__":

    # -------------------------------------------------------------------------------------------------------------
    # The following code solves the problem when you drag the QT Main Window from one screen to another.
    # -------------------------------------------------------------------------------------------------------------
    awareness = ctypes.c_int()
    errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
                    # 0 - DPI unaware - DPI unaware. This app does not scale for DPI changes and is always assumed to have a scale factor of 100% (96 DPI).
                    #     It will be automatically scaled by the system on any other DPI setting.  
                    # 1 - System DPI aware. This app does not scale for DPI changes. It will query for the DPI once and use that value for the
                    #     lifetime of the app. If the DPI changes, the app will not adjust to the new DPI value. It will be automatically scaled
                    #     up or down by the system when the DPI changes from the system value. (Works the best for multiple monitor setups)
                    # 2 - Per monitor DPI aware. This app checks for the DPI when it is created and adjusts the scale factor whenever the DPI changes.
                    #     These applications are not automatically scaled by the system. (does not work too well)
    NewAwareness = 0  # 0 and 1 work well for me
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(NewAwareness)    # Set DPI Awareness  (Windows 10 and 8)

    # -------------------------------------------------------------------------------------------------------------
    # Create the application 
    # -------------------------------------------------------------------------------------------------------------
    # We need one and only one QApplication instance per application
    # This class contains the message queue that handles all event related to the window
    # QApplication(sys.arg) also works and passes 
    app = QApplication([])

    # Create the MyGui instance which has the QMainWindow as a base class
    MyGui = CMyGui()

    # The window is hidden by default, thus show it.
    MyGui.show()

    # The following function sets the top-level widget containing this widget to the the active window.
    # The active window is the top-level window that has the keyboard input focus. This function is likely unnecessary here.
    MyGui.activateWindow()

    # Raises this widget to the top of the parent widget's stack. After this call the widget will be visually in front of
    # any overlapping sibling widgets.
    MyGui.raise_()

    MyGui.resize(1000, 600)

    # app.exec_() start the message queue. If all went well, it returns 0, otherwise some type of error code or message is returned
    # sys.exit(value) return the value to the calling process (usually your shell) 
    Code = app.exec_()
    sys.exit(Code)    