# Filename: FlexLinkConfig.py
# Author:   Andreas Schwarzinger      Date: August 20, 2022
__title__     = "FlexLinkConfig"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Sept, 4rd, 2022"
__copyright__ = 'Andreas Schwarzinger'
__license__   = "MIT"



import math

# --------------------------------------------------------------------
# 1. Choice of oscillator frequency
# --------------------------------------------------------------------
# -> 20.00MHz is a common frequency used for WLAN and is readily available
# -> 20.48MHz is a common frequency an results in a nicer subcarrier spacing
b2048Oscillator = bool(True)
if(b2048Oscillator == True):
    MasterClock = 20.48e6
else:
    MasterClock = 20.00e6

# --------------------------------------------------------------------
# 2. Choice of subcarrier spacing (0 = 20KHz, 1 = 40Khz, 2 = 80KHz, 3 = 160KHz)
# --------------------------------------------------------------------
# -> The subcarrier spacing is defined in terms of MasterClock / 1024
iScIndex = int(1)
assert iScIndex >= 0 and iScIndex < 4, 'The subcarrier index is out of range'
match iScIndex:
    case 0:
        fSubcarrierSpacing = 1 * MasterClock / 1024   # about 20KHz
    case 1:
        fSubcarrierSpacing = 2 * MasterClock / 1024   # about 40KHz
    case 2:
        fSubcarrierSpacing = 4 * MasterClock / 1024   # about 80KHz
    case 3:
        fSubcarrierSpacing = 8 * MasterClock / 1024   # about 160KHz


# ------------------------------------------------------------
# 3. Choice of bandwidth (0 = 5MHz, 1 = 10MHz, 2 = 20MHz, 3 = 40MHz)
# ------------------------------------------------------------
iBwIndex      = int(0)
BandwidthHz   = (MasterClock/4) * 2**iBwIndex  
SampleRateDsp = BandwidthHz
SampleRateAdc = 2 * BandwidthHz
FFT_Size      = BandwidthHz / fSubcarrierSpacing
assert math.fmod(FFT_Size, 1.0) == 0.0, "The FFT Size may not be a fractional value."


# -------------------------------------------------------------------
# 4. Choice of cyclic prefix (0 = 1usec, 1 = 2usec, 2 = 4usec, 3 = 8usec)
# -------------------------------------------------------------------
iCpIndex = int(2)
assert iCpIndex >= 0 and iCpIndex < 4, 'The cyclic prefix index is out of range'
NCpSamples       = 5 * 2**iBwIndex * 2**iCpIndex 
CpLengthSec      = NCpSamples / SampleRateDsp


# ------------------------------------------------------------
print('MasterClock        (Hz) = ' + str(int(MasterClock)))
print('Bandwidth         (MHz) = ' + str(BandwidthHz/1e6))
print('I/FFT Size              = ' + str(int(FFT_Size)))
print('Subcarrier Spacing (Hz) = ' + str(fSubcarrierSpacing))
print('Sample Rate (ADC) (Hz)  = ' + str(int(SampleRateAdc)))
print('Sample Rate (DSP) (Hz)  = ' + str(int(SampleRateDsp)))
print('Number Cp Samples       = ' + str(NCpSamples))
print('Cp Length (sec)         = ' + str(CpLengthSec))
print('IFFT length (sec)       = ' + str( FFT_Size / SampleRateDsp) )
print('OFDM Symbol Length      = ' + str( (FFT_Size + NCpSamples) / SampleRateDsp) )

