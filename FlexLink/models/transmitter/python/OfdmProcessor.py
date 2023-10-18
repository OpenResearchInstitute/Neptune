# File:   OfdmProcessor.py
# Author: Andreas Schwarzinger                                                                 Date: August, 8, 2023
# Notes:  The following script implements OFDM modulation for the Flex Link

__title__     = "OfdmProcessor"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__version__   = "0.1.0.0"
__date__      = "Aug, 8, 2023"
__copyright__ = 'MIT License'
__license__   = "MIT"


# ------------------------------------------------
# Module Imports
# ------------------------------------------------
# was: from   FlexLinkParameters import *
import FlexLinkParameters as fp
import numpy as np



# ------------------------------------------------
# > COfdmProcessor class
# ------------------------------------------------
class COfdmProcessor():
    '''
    brief: This class provides OFDM modulation and demodulation
    '''
    # ---------------------------------
    # > Function: Constructor
    # ---------------------------------
    def __init__( self
                , FlexLinkConfiguration: fp.CFlexLinkConfig):
        # ----------------
        # > Basic Type checking
        # ----------------
        assert isinstance(FlexLinkConfiguration, fp.CFlexLinkConfig)
        
        self.FlexLinkConfiguration = FlexLinkConfiguration




    # ----------------------------------
    # > Function: OfdmModulation
    # ----------------------------------
    def OfdmModulation(self
                     , TxResourceGrid: np.ndarray) ->np.ndarray:
        '''
        brief: This function OFDM modulates the resource grid and generates the transmit waveform
        '''
        # ---------------------
        # > Basic type and error checking
        # ---------------------
        assert isinstance(TxResourceGrid, np.ndarray)
        assert np.issubdtype(TxResourceGrid, np.complexfloating)
        NumSubcarriers, NumOfdmSymbols = TxResourceGrid.size()
        assert NumSubcarriers & 2 == 1, 'Due to the presence of the DC carrier, the NumSubcarriers should be an odd number.'


        N_FFT       = self.FlexLinkConfiguration.FftSize
        NumCpSamples = self.FlexLinkConfiguration.CpDurationSamples

        # -----------------------------------------------------------------------------------------
        # > Determine the size of output waveform array and instantiate it
        # -----------------------------------------------------------------------------------------
        # Iterate through all symbols to be transformed and count all the samples
        NumOutputSamples = NumOfdmSymbols * (NumCpSamples + N_FFT)

        # Instantiate the output waveform
        OutputWaveform = np.zeros([NumOutputSamples], dtype=np.complex64)


        # ----------------------------------------------------------------------------------------
        # > Prepare indexing ahead of IFFT operation
        # ----------------------------------------------------------------------------------------
        # 1. Determine the indices of the positive and negative frequency subcarriers in the input Resource Grid.
        HalfNumSubcarriers = int(np.floor(NumSubcarriers/2))            # As NumSubcarriers is an odd number, the 
                                                                        # HalfNumSubcarriers is a bit of a misnomer
        PosFreqSubcarriers = range(HalfNumSubcarriers, NumSubcarriers)  # Subcarrier Indices of positive frequencies
        NegFreqSubcarriers = range(0, HalfNumSubcarriers)               # Subcarrier Indices of negative frequencies

        # 2. Determine the indices of the positive and negative frequency subcarriers in the IFFT input.
        PosFreqIfftIndices = range(0, HalfNumSubcarriers + 1)           # Ifft indices mapped to positive frequencies
        NegFreqIfftIndices = range(N_FFT - HalfNumSubcarriers, N_FFT)   # and negative frequencies


        # ---------------------------------------------------------------------------------------
        # > Transform the resource grid into the time domain
        # ---------------------------------------------------------------------------------------
        # The OutputSymbolStartIndex indicates where each new Ofdm Symbols starts in the output sequence
        OutputSymbolStartSampleIndex = 0
        for SymbolIndex in range(0, NumOfdmSymbols):
            # Load and compute the IFFT
            IFFT_Input                     = np.zeros([N_FFT], dtype = np.complex64)
            IFFT_Input[PosFreqIfftIndices] = TxResourceGrid[PosFreqSubcarriers, SymbolIndex] 
            IFFT_Input[NegFreqIfftIndices] = TxResourceGrid[NegFreqSubcarriers, SymbolIndex]
            IFFT_Output                    = N_FFT * np.fft.ifft(IFFT_Input, N_FFT)

            # Generate the cyclic prefix, which are the last few samples of the IFFT output
            Cp                             = IFFT_Output[N_FFT - NumCpSamples:N_FFT]

            # Generate the Ofdm symbol in the time domain
            OfdmSymbol                     = np.hstack([Cp, IFFT_Output])

            # Insert the Ofdm Symbol into the Output Waveform Array
            OutputIndices                  = range(OutputSymbolStartSampleIndex,OutputSymbolStartSampleIndex + len(OfdmSymbol))
            OutputWaveform[OutputIndices]  = OfdmSymbol
            OutputSymbolStartSampleIndex  += len(OfdmSymbol)
            
        # -----------------------------------------------------------------------------------
        # > Verify the output waveform length and return it to the calling function.
        # -----------------------------------------------------------------------------------
        assert len(OutputWaveform) == NumOutputSamples, 'The output samples were not properly mapped into the complete Outputwaveform'
        return OutputWaveform





    # ----------------------------------
    # > Function: OfdmDemodulation
    # ----------------------------------
    def OfdmDemodulation(self
                     , RxInputSequence: np.ndarray) -> np.ndarray:
        '''
        brief: This function OFDM demodulates the resource grid and generates RxResourceGrid
        '''
        # ---------------------
        # > Basic type and error checking
        # ---------------------
        assert isinstance(RxInputSequence, np.ndarray)
        assert np.issubdtype(RxInputSequence, np.complexfloating)