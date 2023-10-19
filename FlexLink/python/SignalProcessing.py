# File:     SignalProcessing.py
# Notes:    This file provides basic signal processing functions
#           1. AddAdwgn()     - This function adds Additive White Gaussian Noise to a real or complex sequence of numbers
#           2. AddMultipath() - This function adds multiple time domain paths of a complex transmit sequence.


__title__     = "SignalProcessing"
__author__    = "Andreas Schwarzinger"
__status__    = "released"
__date__      = "Oct, 28rd, 2022"
__copyright__ = 'Andreas Schwarzinger'

import numpy             as np
import matplotlib.pyplot as plt
import math

# --------------------------------------------------------------
# > AddAwgn()
# --------------------------------------------------------------
def AddAwgn(SnrdB:          float 
          , InputSequence:  np.ndarray) -> np.ndarray:
    """
    brief:  This function will add white Gaussian noise to a real or complex sequence in a one dimensional numpy array.
    param:  SnrdB         - The signal to noise ratio of the noise sequence to be generated
    param:  InputSequence - A numpy ndarray of real or complex sequence of numbers 
    return: A new array is returned that contains the sum of noise and the input sequence.
    """

    # -----------------------------------------
    # Error checking
    assert isinstance(SnrdB, float) or isinstance(SnrdB, int), 'The SnrdB input argument must be a numeric type.'
    assert isinstance(InputSequence, np.ndarray),              'The InputSequence must be a numpy array.'
    assert np.issubdtype(InputSequence.dtype, np.number),      'The InputSequence entries must be real or complex numbers.'
    assert len(InputSequence.shape) == 1,                      'The InputSequence must be a simple one dimensional array.'

    # -----------------------------------------
    # We need slighlty different code for real and complexfloat types
    IsComplex = np.issubdtype(InputSequence.dtype, np.complexfloating)

    # Convert SNR in dB to linear
    SnrLinear = 10 ** (SnrdB/10)

    # Compute the signal power (mean square of the input sequence)
    N = len(InputSequence)
    if IsComplex == True:
        MeanSquareSignal = (1/N) * np.sum(InputSequence * np.conj(InputSequence))
    else:
        MeanSquareSignal = (1/N) * np.sum(InputSequence * InputSequence)

    # Compute the required noise power (mean square of the noise sequence)
    MeanSquareNoise = MeanSquareSignal / SnrLinear

    if IsComplex == True:
        Scalar        = np.sqrt(MeanSquareNoise) * np.sqrt(1/2)
        NoiseSequence = Scalar * np.array(np.random.randn(N) + 1j*np.random.randn(N), dtype = InputSequence.dtype)  
    else:
        NoiseSequence = np.sqrt(MeanSquareNoise) * np.array(np.random.randn(N), dtype = np.float32)  

    # Return noisy input sequency
    return InputSequence + NoiseSequence

    






# --------------------------------------------------------------
# > AddMultipath()
# --------------------------------------------------------------
def AddMultipath(InputSequence:  np.ndarray
               , SampleRate:     float
               , Delays:         list
               , Constants:      list
               , Dopplers:       list) -> tuple:
    """
    Brief: This function will add multipath distortion to the input signal. 
           Note that the entries in the Delays list are integers representing sample delay.
    param: InputSequence  I  A numpy array with complex floating point entries
    param: SampleRate     I  An integer or floating point scalar
    param: Delays         I  A list of integer sample delays
    param: Constants      I  A list of complex values that scale the waveform
    param: Dopplers       I  A list of integer or floating point Doppler frequencies
    """

    # -----------------------------------------
    # Error checking
    assert isinstance(InputSequence, np.ndarray),                             'The InputSequence must be a numpy array.'
    assert np.issubdtype(InputSequence.dtype, np.complexfloating),            'The InputSequence entries must be complex floating point numbers.'
    assert len(InputSequence.shape) == 1,                                     'The InputSequence must be a simple one dimensional array.'
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int),      'The SampleRate input argument must be a numeric type.'
    assert isinstance(Delays, list) and isinstance(Delays[0], int),           'The Delays argument must be a list of integers.'
    assert isinstance(Constants, list) and isinstance(Constants[0], complex), 'The Constants argument must be a list of complex value.'
    assert isinstance(Dopplers, list),                                        'The Doppler argument must be a list of floating point values.' 
    assert isinstance(Dopplers[0], int) or isinstance(Dopplers[0], float),    'The list of doppler values must be either integer or floating point based.'
    assert len(Delays) == len(Constants) and len(Delays) == len(Dopplers),    'The length of Delays, Constants, and Doppler lists must be the same.'

    # -----------------------------------------
    # Determine the delay range and length of the input sequence
    N             = len(InputSequence)
    DType         = InputSequence.dtype
    n             = np.arange(0, N)
    DelayDistance = max(Delays) - min(Delays) 
    M             = N + DelayDistance    # The length of the output waveform

    # Allocate memory for the output waveform
    OutputSequence = np.zeros(M, dtype = DType)

    # Generate each path and add it to the output sequence
    NumberOfPaths = len(Delays)
    for PathIndex in range(0, NumberOfPaths):
        Delay      = Delays[PathIndex]
        Constant   = Constants[PathIndex]
        Doppler    = Dopplers[PathIndex] 

        # Compute the start index in the output sequence
        StartIndex      = Delay - min(Delays)
        OutputSequence[StartIndex:StartIndex + N] += \
                Constant * np.exp(1j*2*np.pi*Doppler*n/SampleRate, dtype = DType) * InputSequence

    # Return the output sequence
    return OutputSequence, min(Delays)

    


# --------------------------------------------------------------- #
# > SpectrumAnalyzer()
# --------------------------------------------------------------- # 
def SpectrumAnalyzer( InputSequence: np.ndarray
                    , SampleRate     
                    , FFT_Size:      int
                    , bPlot:         bool):
    '''
    brief: This function computes the power spectrum of a real or complex sequence
    param: InputSequence - A 1D numpy array representing the sequence to be analyzed
    param: FFT_Size      - The FFT size (Resolution bandwidth = SampleRate/FFT_Size)
    param: bPlot         - A boolean indicating whether to plot the power spectrum
    '''

    # ----------------------------------------------
    # Type checking
    assert isinstance(InputSequence, np.ndarray)
    assert isinstance(SampleRate, float) or isinstance(SampleRate, int)
    assert isinstance(FFT_Size, int)
    assert isinstance(bPlot, bool)

    # Recast the input sequence to a complex type
    InputSequence = InputSequence.astype(np.complex128)

    # ----------------------------------------------
    # Determine the quantities N = FFT_Size, M, and P
    # M is the number of samples in one IQ sections 
    # N is the number of samples in one IQ subsections
    # R is the number of subsections in one full IQ section
    IqLength = len(InputSequence) 
    N        = FFT_Size
    MinR     = 4
    assert IqLength >= N * MinR, 'The number of IQ samples must be >= FFT_Size * 4.'

    R        = MinR
    if IqLength >= N*8:   R = 8
    if IqLength >= N*16:  R = 16
    if IqLength >= N*32:  R = 32

    M  = N * R     # M is the number of IQ sample in one section of the InputSequence 
    nn = np.arange(0, M, 1, np.int32)
    k  = np.arange(-np.floor(M/2), np.floor(M/2), 1, np.int32) 

    # ----------------------------------------------
    # Build the desired window
    NumSinusoids   = R + 1
    Ak             = np.ones(NumSinusoids, np.float32)
    Ak[0] = Ak[-1] = np.float32(0.91) # 416)

    Sinusoids      = np.zeros(M, np.complex128)
    for Index in range(0, NumSinusoids):
        f          = ((-R/2) + Index)/M
        Sinusoids += Ak[Index] * np.exp(1j*2*np.pi*k*f)

    Hanning       = 0.5 - 0.5 * np.cos(2*np.pi*(nn + 1) / (M + 1))
    DesiredWindow = Sinusoids * Hanning

    # ----------------------------------------------
    # Run through each iteration (section of M samples)
    # We want to use all samples in the IqSequence to compute the power spectrum.
    # Determine how many sections of the M samples are available for spectrum analysis
    NumSections   = int(np.ceil(IqLength/M))
    PowerSpectrum = np.zeros(N, np.complex128)
    for Section in range(0, NumSections): 
        if   Section == 0:
            IqMSequence = InputSequence[0:M]
        elif Section == NumSections-1:
            IqMSequence = InputSequence[(-M-1):-1]    
        else:
            StartPosition = Section * math.floor(IqLength / NumSections)
            StopPosition  = StartPosition + M
            IqMSequence   = InputSequence[StartPosition:StopPosition]
    
        IqMWindowed   = IqMSequence * DesiredWindow

        # -----------------------------------------------
        # Break up each section into R N-sized subsections and add them up
        IqNSequence = np.zeros(N, np.complex128)
        for Subsection in range(0, R):
            StartPosition = Subsection * N
            StopPosition  = StartPosition + N
            IqNSequence  += (1/R) * IqMWindowed[StartPosition:StopPosition]

        # ----------------------------------------------
        # Take the FFT
        Fft            = (1/N) * np.fft.fft(IqNSequence)
        PowerSpectrum += (1/NumSections) * (Fft * np.conj(Fft))

    # Rearrange the power spectrum such that the negative frequencies appear first
    PowerSpectrum = np.roll(PowerSpectrum, math.floor(N/2))
    Frequencies   = np.arange(-0.5*SampleRate, 0.5*SampleRate - 1e-6, SampleRate/N, \
                                                                           np.float32)
    ResolutionBW  = float(SampleRate / N)

    if bPlot == True:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(Frequencies, np.abs(PowerSpectrum), 'k')
        plt.grid(True)
        plt.title('Linear Power Spectrum')
        plt.xlabel('Hz')
        plt.tight_layout()
        plt.subplot(2,1,2)
        plt.plot(Frequencies, 10*np.log10(np.abs(PowerSpectrum)), 'k')
        plt.grid(True)
        plt.title('Power Spectrum in dB')
        plt.xlabel('Hz')
        plt.ylabel('dB')
        plt.tight_layout()
        plt.show()

    return PowerSpectrum, Frequencies, ResolutionBW





# ------------------------------------------------------------
# > Test bench
# ------------------------------------------------------------
if __name__ == '__main__':
    Test = 2

    if Test == 1:
        InputSequence = np.ones(20, dtype = np.complex64)
        Delays        = [-1, 1, 2]
        Constants     = [1 + 0j, 0, -0.2j]
        Dopplers      = [1000, 0, 5000]
        SampleRate    = 100000
        Output, MinDelay = AddMultipath(InputSequence, SampleRate, Delays, Constants, Dopplers)

        print('MinSampleDelay: ' + str(MinDelay))
        print('MinTimeDelay:   ' + str(MinDelay / SampleRate))

        plt.figure(1)
        plt.plot( Output.real, c = 'red')
        plt.plot( Output.imag, c = 'blue')
        plt.grid(True)
        plt.show()
        stop = 1  # Example call

    if Test == 2:
        SampleRate      = 1
        TotalNumSamples = 64
        N               = 8
        n               = np.arange(0, TotalNumSamples, 1)
        IqSequence0     = np.exp(1j*2*np.pi*n*0.0625/SampleRate)
        IqSequence1     = np.exp(1j*2*np.pi*n*0.07/SampleRate)
        IqSequence2     = np.exp(1j*2*np.pi*n*0.1/SampleRate)
        MeanSquare      = np.mean(IqSequence0 * np.conj(IqSequence0)).real

        print('Total Power:          ' + str(MeanSquare))

        PowerSpectrum0, Frequencies, ResolutionBW = SpectrumAnalyzer( InputSequence = IqSequence0
                                                                    , SampleRate    = SampleRate
                                                                    , FFT_Size      = N
                                                                    , bPlot         = False)
        print('Total Power Detected: ' + str(np.sum(PowerSpectrum0).real))


        PowerSpectrum1, Frequencies, ResolutionBW = SpectrumAnalyzer( InputSequence = IqSequence1
                                                                    , SampleRate    = SampleRate
                                                                    , FFT_Size      = N
                                                                    , bPlot         = False)
        print('Total Power Detected: ' + str(np.sum(PowerSpectrum1).real))


        PowerSpectrum2, Frequencies, ResolutionBW = SpectrumAnalyzer( InputSequence = IqSequence2
                                                                    , SampleRate    = SampleRate
                                                                    , FFT_Size      = N
                                                                    , bPlot         = False)
        print('Total Power Detected: ' + str(np.sum(PowerSpectrum2).real))

        print('Resolution Bandwidth: ' + str(ResolutionBW) + ' Hz')
        