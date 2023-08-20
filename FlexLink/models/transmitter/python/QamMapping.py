# File:  QamMapping.py
# Notes: This file provides both QAM Mapping and demapping as described in IEEE Std: 802.11-2012

# Imports
import numpy            as     np
from   SignalProcessing import *

# Check: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting

__title__     = "QamMapping"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Oct, 3rd, 2022"
__copyright__ = 'Andreas Schwarzinger'



# ------------------------------------------------------------------------------------------------
# > Definition of the CQamMapping
# ------------------------------------------------------------------------------------------------
class CQamMappingIEEE():
    """
    This class provides QAM mapping and demapping as described in IEEE Std: 802.11-2012
    """

    # -----------------------------------------------
    # Definition of QAM Constellations 
    BPSK_LUT  = np.array([-1, 1])                                                             # Table 18.8 in the specifications
    QPSK_LUT  = (1/np.sqrt(2))  * np.array([-1.0, 1.0],                  dtype = np.float32)  # Table 18.9 in the specifications
    QAM16_LUT = (1/np.sqrt(10)) * np.array([-3, -1, 3, 1],               dtype = np.float32)  # Table 18.10 in the specifications 
    QAM64_LUT = (1/np.sqrt(42)) * np.array([-7, -5, -1, -3, 7, 5, 1, 3], dtype = np.float32)  # Table 18.11 in the specifications

    QAM16_Table = {-3: [0,0], \
                   -1: [0,1], \
                    1: [1,1], \
                    3: [1,0]  }

    QAM64_Table = {-7: [0,0,0], \
                   -5: [0,0,1], \
                   -3: [0,1,1], \
                   -1: [0,1,0], \
                    1: [1,1,0], \
                    3: [1,1,1], \
                    5: [1,0,1], \
                    7: [1,0,0]  }   



    # -----------------------------------------------------
    # > Function -> CQamMappingIEEE.Mapping()
    # -----------------------------------------------------
    @classmethod
    def Mapping(cls
              , BitsPerSymbol:  int
              , InputBits:      np.ndarray):
        """
        This function provides BPSK, QPSK, 16QAM and 64QAM mapping
        """
        assert isinstance(BitsPerSymbol, int),                  'The BitPerSymbol input argument must be an integer.'
        assert any([x == BitsPerSymbol for x in [1, 2, 4, 6]]), 'The BitPerSymbol input argument must be either 1, 2, 4 or 6.'
        assert isinstance(InputBits, np.ndarray),               'The Input Bit vector must be an numpy array'
        assert np.issubdtype(InputBits.dtype, np.integer),      'The input bit vector must contain numpy.int8 type entries'
        NumInputBits = len(InputBits)
        assert NumInputBits % BitsPerSymbol == 0,                'The input bits vector is of invalid size'

        # Check that these are really bits
        for InputBit in InputBits:
            assert InputBit == 0 or InputBit == 1, 'The input bit must be 0 or 1.'

        # Start the mapping process
        NumQamSymbols  = int(NumInputBits / BitsPerSymbol)
        QamSymbols     = np.zeros(NumQamSymbols, dtype = np.complex64)
        InputBitIndex  = 0
        QamSymbolIndex = 0
        while InputBitIndex < NumInputBits:
            # Process BPSK
            if(BitsPerSymbol == 1):
                QamTableOffset = InputBits[InputBitIndex]
                QamSymbols[QamSymbolIndex] = cls.BPSK_LUT[QamTableOffset] + 0j
                InputBitIndex  += 1
                QamSymbolIndex += 1
                continue

            # Process QPSK
            if(BitsPerSymbol == 2):
                QamTableOffsetI = InputBits[InputBitIndex]
                QamTableOffsetQ = InputBits[InputBitIndex + 1]
                I = cls.QPSK_LUT[QamTableOffsetI]
                Q = cls.QPSK_LUT[QamTableOffsetQ]
                QamSymbols[QamSymbolIndex] = np.complex64(I + 1j*Q)
                InputBitIndex  += 2
                QamSymbolIndex += 1
                continue

            # Process 16QAM
            if(BitsPerSymbol == 4):
                QamTableOffsetI = 2 * InputBits[InputBitIndex + 0] + \
                                      InputBits[InputBitIndex + 1]
                QamTableOffsetQ = 2 * InputBits[InputBitIndex + 2] + \
                                      InputBits[InputBitIndex + 3] 
                I = cls.QAM16_LUT[QamTableOffsetI]
                Q = cls.QAM16_LUT[QamTableOffsetQ]
                QamSymbols[QamSymbolIndex] = np.complex64(I + 1j*Q)
                InputBitIndex  += 4
                QamSymbolIndex += 1
                continue

            # Process 64QAM
            if(BitsPerSymbol == 6):
                QamTableOffsetI = 4 * InputBits[InputBitIndex + 0] + \
                                  2 * InputBits[InputBitIndex + 1] + \
                                  1 * InputBits[InputBitIndex + 2]
                QamTableOffsetQ = 4 * InputBits[InputBitIndex + 3] + \
                                  2 * InputBits[InputBitIndex + 4] + \
                                  1 * InputBits[InputBitIndex + 5]
                I = cls.QAM64_LUT[QamTableOffsetI]
                Q = cls.QAM64_LUT[QamTableOffsetQ]
                QamSymbols[QamSymbolIndex] = np.complex64(I + 1j*Q)
                InputBitIndex  += 6
                QamSymbolIndex += 1
                continue

        return QamSymbols




    # -----------------------------------------------------
    # > Function -> CQamMappingIEEE.HardDemapping()
    # -----------------------------------------------------
    @classmethod
    def HardDemapping(cls
                    , BitsPerSymbol:  int
                    , Estimate_x:     np.complex64) -> np.ndarray:
        """
        brief: This function provides BPSK, QPSK, 16QAM and 64QAM hard demapping
        param: BitsPerSymbol  - 1/2/4/6 for BPSK/QPSK/16QAM/64QAM
        param: Estimate_x     - The equalized observation Y
        """
        assert isinstance(BitsPerSymbol, int),                      'The BitPerSymbol input argument must be an integer.'
        assert any([x == BitsPerSymbol for x in [1, 2, 4, 6]]),     'The BitPerSymbol input argument must be either 1, 2, 4 or 6.'
        assert isinstance(Estimate_x, np.ndarray),                  'Invalid type of Estimate_x'
        assert np.issubdtype(Estimate_x.dtype, np.complexfloating), 'Invalid type of Estimate_x[0]'

        NumInputSymbols  = len(Estimate_x)
        NumOutputBits    = NumInputSymbols * BitsPerSymbol
        OutputBits       = np.zeros(NumOutputBits, dtype = np.int8)

        BitIndex = 0
        for Estimate in Estimate_x:
            if(BitsPerSymbol == 1):
                if Estimate.real > 0: 
                    OutputBits[BitIndex] = 1
                BitIndex            += 1

            if(BitsPerSymbol == 2):
                if Estimate.real > 0: 
                    OutputBits[BitIndex] = 1
                BitIndex            += 1
                if Estimate.imag > 0: 
                    OutputBits[BitIndex] = 1
                BitIndex            += 1

            if(BitsPerSymbol == 4):
                Real = Estimate.real
                Imag = Estimate.imag
                if(Real >  1): Real =  1
                if(Real < -1): Real = -1
                if(Imag >  1): Imag =  1
                if(Imag < -1): Imag = -1
                Real = 2 * (np.floor(Real * np.sqrt(10)/2) + 0.5)
                Imag = 2 * (np.floor(Imag * np.sqrt(10)/2) + 0.5)  
                OutputBits[BitIndex+0:BitIndex+2] = cls.QAM16_Table[Real]
                OutputBits[BitIndex+2:BitIndex+4] = cls.QAM16_Table[Imag]
                BitIndex += 4

                
            if(BitsPerSymbol == 6):
                Real = Estimate.real
                Imag = Estimate.imag
                if(Real >  1.1): Real =  1.1
                if(Real < -1.1): Real = -1.1
                if(Imag >  1.1): Imag =  1.1
                if(Imag < -1.1): Imag = -1.1
                Real = 2 * (np.floor(Real * np.sqrt(42)/2) + 0.5)
                Imag = 2 * (np.floor(Imag * np.sqrt(42)/2) + 0.5)           
                OutputBits[BitIndex+0:BitIndex+3] = cls.QAM64_Table[Real]
                OutputBits[BitIndex+3:BitIndex+6] = cls.QAM64_Table[Imag]
                BitIndex += 6

        return OutputBits




    # -----------------------------------------------------
    # > Function -> CQamMappingIEEE.SoftDemapping()
    # -----------------------------------------------------
    @classmethod
    def SoftDemapping(cls
                    , BitsPerSymbol:  int
                    , Estimate_x:     np.complex64
                    , NoiseVariance:  float
                    , h) -> np.ndarray:
        """
        brief: This function provides BPSK, QPSK, 16QAM and 64QAM soft demapping using LLR
        param: BitsPerSymbol - 1/2/4/6 for BPSK/QPSK/16QAM/64QAM
        param: Estimate_x    - The equalized observation Y
        param: NoiseVariance - Noise variance of n at the time of observation
        param: h             - The frequency response for the resource element associated with this QAM symbol
        """
        assert isinstance(BitsPerSymbol, int),                      'The BitPerSymbol input argument must be an integer.'
        assert any([x == BitsPerSymbol for x in [1, 2, 4, 6]]),     'The BitPerSymbol input argument must be either 1, 2, 4 or 6.'
        assert isinstance(Estimate_x, np.complex64),                  'Invalid type of Estimate_x'
        assert np.issubdtype(Estimate_x.dtype, np.complexfloating), 'Invalid type of Estimate_x[0]'
        assert np.issubdtype(h.dtype, np.complexfloating),          'The frequency response h must be complex'

        if(BitsPerSymbol == 1):
            y_I     = Estimate_x.real
            LlrBits = np.array( (4 * (h*h.conj()).real / NoiseVariance) * y_I, dtype=np.float32)

        if(BitsPerSymbol == 2):
            y_I     = Estimate_x.real
            y_Q     = Estimate_x.imag
            LlrBits = np.array([(4 * (h*h.conj()).real / (0.7071 * NoiseVariance)) * y_I, \
                                (4 * (h*h.conj()).real / (0.7071 * NoiseVariance)) * y_Q], dtype = np.float32)

        if(BitsPerSymbol == 4):
            LlrBits    = np.zeros(4, dtype = np.float32)
            C          = np.float32(1/ np.sqrt(10))
            y_I        = Estimate_x.real
            y_Q        = Estimate_x.imag
            LlrBits[0] = cls.GetLLR([ 3*C,  1*C], [-3*C,   -C], y_I, NoiseVariance, h)         
            LlrBits[1] = cls.GetLLR([-1*C,  1*C], [-3*C,  3*C], y_I, NoiseVariance, h)  
            LlrBits[2] = cls.GetLLR([ 3*C,  1*C], [-3*C,   -C], y_Q, NoiseVariance, h)  
            LlrBits[3] = cls.GetLLR([-1*C,  1*C], [-3*C,  3*C], y_Q, NoiseVariance, h)  

    
        if(BitsPerSymbol == 6):
            LlrBits    = np.zeros(6, dtype = np.float32)
            C          = np.float32(1/ np.sqrt(42))
            y_I        = Estimate_x.real
            y_Q        = Estimate_x.imag
            LlrBits[0] = cls.GetLLR([ 7*C,  5*C,  3*C,  1*C], [-7*C, -5*C, -3*C, -1*C], y_I, NoiseVariance, h)         
            LlrBits[1] = cls.GetLLR([-3*C, -1*C,  3*C,  1*C], [-7*C, -5*C,  5*C,  7*C], y_I, NoiseVariance, h)  
            LlrBits[2] = cls.GetLLR([-5*C, -3*C,  3*C,  5*C], [-7*C, -1*C,  1*C,  7*C], y_I, NoiseVariance, h)  
            LlrBits[3] = cls.GetLLR([ 7*C,  5*C,  3*C,  1*C], [-7*C, -5*C, -3*C, -1*C], y_Q, NoiseVariance, h)        
            LlrBits[4] = cls.GetLLR([-3*C, -1*C,  3*C,  1*C], [-7*C, -5*C,  5*C,  7*C], y_Q, NoiseVariance, h)  
            LlrBits[5] = cls.GetLLR([-5*C, -3*C,  3*C,  5*C], [-7*C, -1*C,  1*C,  7*C], y_Q, NoiseVariance, h)             

        return LlrBits



    # -------------------------------------------------------
    # > Function -> CQamMappingIEEE.GetLLR()
    # -------------------------------------------------------
    @staticmethod
    def GetLLR(Position1:      list
             , Position0:      list
             , y:              np.float32
             , NoiseVariance:  float
             , h:              np.complex64) -> np.float32:
        """
        brief: This function provides the log likelihood ratio for a particular bit
        reference: Digital Signal Processing in Modern Communication Systems (Edition 2) Section 5.6.2
        notes: Y         = observation = h*x + n
        notes: est(x)    = Y/h = x + n/h
        param: Position1 - A set of N means that belong to N Gaussian PDFs. They are related to the bit 1
        param: Position0 - A set of N means that belong to N Gaussian PDFs. They are related to the bit 0
        param: y         - real(est(x)) or imag(est(x))
        param: NoiseVariance - NoiseVariance of n at the time of observation
        param: h         - The frequency response for the resource element associated with this QAM symbol
        """
        assert type(Position1) == list,                                           'The type of Position1 is invalid'
        assert type(Position0) == list,                                           'The type of Position0 is invalid'
        assert len(Position1)  == len(Position0),                                 'The number of positions must be the same'
        assert type(y) == np.float32,                                             'The observation y must be complex'
        assert type(NoiseVariance) == np.float32 or type(NoiseVariance) == float, 'The noise variance must be a floating point number'
        assert type(h) == np.complex64 or type(h) == np.complex128,               'The frequency response h must be complex'

        Numerator             = np.float32(1e-10)
        Denominator           = np.float32(1e-10)
        NoiseVarianceEstimate = NoiseVariance / (h * h.conj()).real
        Const1                = 1 / np.sqrt(6.2831853*NoiseVarianceEstimate)
        Const2                = 2*NoiseVarianceEstimate

        # We now build the ratio numerator/denominator of which we will take the log
        for Index, position1 in enumerate(Position1):
            position0    = Position0[Index]
            Numerator   += Const1 * np.exp(-((y-position1)**2)/Const2)
            Denominator += Const1 * np.exp(-((y-position0)**2)/Const2)

        LLRdB = np.log(Numerator/Denominator)
        return LLRdB













# ---------------------------------------------------------
# Function: The test bench
# ---------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    os.system('cls')  # This statement resets the terminal window

    # Construct the CQamMappingIEEE object
    Mapper = CQamMappingIEEE()

    # Select the test you want to run
    TestSelection = {'MapperView'               : False,  \
                     'MapperTestSimple'         : False, \
                     'MapperTestRandom'         : False, \
                     'SoftDemappingSimple'      : False, \
                     'SoftDemappingSimpleBPSK'  : False, \
                     'SoftDemappingSimpleQPSK'  : False, \
                     'SoftDemappingSimple16QAM' : False, \
                     'SoftDemappingSimple64QAM' : False, \
                     'BerTest'                  : True }

    NumInputBits = 24000
    rng          = np.random.default_rng(12345)      
    InputBits    = np.array(rng.integers(low=0, high=2, size=NumInputBits), dtype = np.uint8)
    

    # --------------------------------------------------------
    # MapperView  
    # --------------------------------------------------------
    if(TestSelection['MapperView'] == True):
        BPSKSymbols   = Mapper.Mapping(1, InputBits)
        QPSKSymbols   = Mapper.Mapping(2, InputBits)
        Qam16Symbols   = Mapper.Mapping(4, InputBits)
        Qam64Symbols   = Mapper.Mapping(6, InputBits)

        CinrdB    = 22
        BPSKSymbolsNoisy  = AddAwgn(CinrdB, BPSKSymbols)
        QPSKSymbolsNoisy  = AddAwgn(CinrdB, QPSKSymbols)
        Qam16SymbolsNoisy = AddAwgn(CinrdB, Qam16Symbols)
        Qam64SymbolsNoisy = AddAwgn(CinrdB, Qam64Symbols)

        Const1 = 1/np.sqrt(10)
        Const2 = 1/np.sqrt(42)

        plt.figure(1)
        plt.subplot(2,2,1)
        plt.plot(BPSKSymbolsNoisy.real, BPSKSymbolsNoisy.imag, 'r.')
        plt.grid(True)
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.tight_layout()
        plt.title('BPSK Constellation')
        plt.subplot(2,2,2)
        plt.plot(QPSKSymbolsNoisy.real, QPSKSymbolsNoisy.imag, 'r.')
        plt.grid(True)
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.tight_layout()
        plt.title('QPSK Constellation')
        plt.subplot(2,2,3)
        plt.plot(Qam16SymbolsNoisy.real, Qam16SymbolsNoisy.imag, 'r.')
        plt.grid(True)
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.title('16QAM Constellation')
        plt.xticks([-2*Const1, 0, 2*Const1])
        plt.yticks([-2*Const1, 0, 2*Const1])
        plt.tight_layout()
        plt.subplot(2,2,4)
        plt.plot(Qam64SymbolsNoisy.real, Qam64SymbolsNoisy.imag, 'r.')
        plt.grid(True)
        plt.xticks([-6*Const2, -4*Const2, -2*Const2, 0, 2*Const2, 4*Const2, 6*Const2])
        plt.yticks([-6*Const2, -4*Const2, -2*Const2, 0, 2*Const2, 4*Const2, 6*Const2])
        plt.axis([-1.2, 1.2, -1.2, 1.2])
        plt.title('64QAM Constellation')
        plt.tight_layout()


    # --------------------------------------------------------
    # MapperTestSimple  
    # --------------------------------------------------------
    if(TestSelection['MapperTestSimple'] == True):
        BitsPerSymbol = 1   # BPSK
        BPSKSymbols   = Mapper.Mapping(BitsPerSymbol, np.array([0,1], dtype=np.uint8))
        Bits          = Mapper.HardDemapping(BitsPerSymbol, BPSKSymbols)
        assert all(Bits == [0, 1]), 'Error'
        
        BitsPerSymbol = 2   # QPSK
        QPSKSymbols = Mapper.Mapping(BitsPerSymbol, np.array([0, 0], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, QPSKSymbols)
        assert all(Bits == [0, 0]), 'Error'
        QPSKSymbols = Mapper.Mapping(BitsPerSymbol, np.array([1, 1], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, QPSKSymbols)
        assert all(Bits == [1, 1]), 'Error'

        BitsPerSymbol = 4   # 16QAM
        Qam16Symbol0 = Mapper.Mapping(BitsPerSymbol, np.array([0, 0,  0, 0 ], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam16Symbol0)
        assert all(Bits == [0, 0,  0, 0 ]), 'Error'
        Qam16Symbol1 = Mapper.Mapping(BitsPerSymbol, np.array([0, 1,  0, 1 ], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam16Symbol1)
        assert all(Bits == [0, 1,  0, 1 ]), 'Error'
        Qam16Symbol2 = Mapper.Mapping(BitsPerSymbol, np.array([1, 1,  1, 1 ], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam16Symbol2)
        assert all(Bits == [1, 1,  1, 1 ]), 'Error'
        Qam16Symbol3 = Mapper.Mapping(BitsPerSymbol, np.array([1, 0,  1, 0 ], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam16Symbol3)
        assert all(Bits == [1, 0,  1, 0 ]), 'Error'

        BitsPerSymbol = 6   # 64QAM
        Qam64Symbol0 = Mapper.Mapping(BitsPerSymbol, np.array([0, 0, 0,  0, 0, 0], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam64Symbol0)
        assert all(Bits == [0, 0, 0,  0, 0, 0]), 'Error'
        Qam64Symbol1 = Mapper.Mapping(BitsPerSymbol, np.array([0, 0, 1,  0, 0, 1], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam64Symbol1)
        assert all(Bits == [0, 0, 1,  0, 0, 1]), 'Error'
        Qam64Symbol2 = Mapper.Mapping(BitsPerSymbol, np.array([0, 1, 1,  0, 1, 1], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam64Symbol2)
        assert all(Bits == [0, 1, 1,  0, 1, 1]), 'Error'
        Qam64Symbol3 = Mapper.Mapping(BitsPerSymbol, np.array([0, 1, 0,  0, 1, 0], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam64Symbol3)
        assert all(Bits == [0, 1, 0,  0, 1, 0]), 'Error'
        Qam64Symbol4 = Mapper.Mapping(BitsPerSymbol, np.array([1, 1, 0,  1, 1, 0], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam64Symbol4)
        assert all(Bits == [1, 1, 0,  1, 1, 0],), 'Error'
        Qam64Symbol5 = Mapper.Mapping(BitsPerSymbol, np.array([1, 1, 1,  1, 1, 1], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam64Symbol5)
        assert all(Bits == [1, 1, 1,  1, 1, 1]), 'Error'
        Qam64Symbol6 = Mapper.Mapping(BitsPerSymbol, np.array([1, 0, 1,  1, 0, 1], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam64Symbol6)
        assert all(Bits == [1, 0, 1,  1, 0, 1]), 'Error'
        Qam64Symbol7 = Mapper.Mapping(BitsPerSymbol, np.array([1, 0, 0,  1, 0, 0], dtype=np.uint8))
        Bits        = Mapper.HardDemapping(BitsPerSymbol, Qam64Symbol7)
        assert all(Bits == [1, 0, 0,  1, 0, 0]), 'Error'

        print('MapperTestSimple has passed')
        

    # --------------------------------------------------------
    # MapperTestRandom  
    # --------------------------------------------------------
    if(TestSelection['MapperTestRandom'] == True):
        BitsPerSymbol = 1   # BPSK
        BPSKSymbols   = Mapper.Mapping(BitsPerSymbol, InputBits)
        Bits          = Mapper.HardDemapping(BitsPerSymbol, BPSKSymbols)
        assert all(Bits == InputBits), 'Error'
            
        BitsPerSymbol = 2   # QPSK
        QPSKSymbols   = Mapper.Mapping(BitsPerSymbol, InputBits)
        Bits          = Mapper.HardDemapping(BitsPerSymbol, QPSKSymbols)
        assert all(Bits == InputBits), 'Error'


        BitsPerSymbol = 4   # 16QAM
        Qam16Symbol0  = Mapper.Mapping(BitsPerSymbol, InputBits)
        Bits          = Mapper.HardDemapping(BitsPerSymbol, Qam16Symbol0)
        assert all(Bits == InputBits), 'Error'
           

        BitsPerSymbol = 6   # 64QAM
        Qam64Symbol0  = Mapper.Mapping(BitsPerSymbol, InputBits)
        Bits          = Mapper.HardDemapping(BitsPerSymbol, Qam64Symbol0)
        assert all(Bits == InputBits), 'Error'
            
        print('MapperTestRandom has passed')



    # --------------------------------------------------------
    # SoftDemappingSimple Test
    # --------------------------------------------------------
    if(TestSelection['SoftDemappingSimple'] == True):  
        BitsPerSymbol = 1   # BPSK
        FreqResponse  = np.complex64(2 + 3j)
        NoiseVariance = np.float32(2)
        BPSKSymbols   = Mapper.Mapping(BitsPerSymbol, np.array([0], dtype=np.uint8))

        # The observation before white Gaussian noise
        Y             = np.complex64(BPSKSymbols[0] * FreqResponse)
        
        ComplexNoise  = np.complex64( 0.7071*np.random.normal(0, np.sqrt(NoiseVariance) ) + \
                                   1j*0.7071*np.random.normal(0, np.sqrt(NoiseVariance) ))
        
        # The observation after adding white Gaussian noise
        Y = Y + ComplexNoise

        # The noise estimate of the QAM symbol
        Estimate = Y / FreqResponse

        LLRBit = Mapper.SoftDemapping(BitsPerSymbol
                                    , Estimate
                                    , NoiseVariance 
                                    , FreqResponse)




    # --------------------------------------------------------
    # SoftDemappingSimpleBPSK
    # --------------------------------------------------------
    # The following test bench comes from the MatLab 'LlrDemap_TestBench2.m'
    # Digital Signal Processing in Modern Communication Systems (Edition 2): Chapter 5 - Section 6: 
    if(TestSelection['SoftDemappingSimpleBPSK'] == True): 
        BitsPerSymbol  = 1   # BPSK
        TxSymbolReal   = np.arange(-1.2, 1.2, 0.01, dtype = np.float32)
        TxSymbolImag   = TxSymbolReal
        TxSymbols      = np.complex64(TxSymbolReal + 1j*TxSymbolImag)
         
        FreqResponse   = np.complex64(1)
        NoiseVariance  = np.float32(0.1)

        NumberOfIterations = len(TxSymbols)
        LLRBits = np.zeros([NumberOfIterations, BitsPerSymbol], dtype = np.float32)

        for SymbolIndex, Symbol in enumerate(TxSymbols):
            LLRBits[SymbolIndex, :] = Mapper.SoftDemapping(BitsPerSymbol
                                                         , Symbol
                                                         , NoiseVariance 
                                                         , FreqResponse)

        plt.figure(1)
        plt.plot(TxSymbolReal, LLRBits[:, 0], c = 'red')
        plt.ylabel('LLR dB')
        plt.xlabel('Real Part of Estimate')
        plt.legend(['bit0'])
        plt.grid(True)
         

    # --------------------------------------------------------
    # SoftDemappingSimpleQPSK
    # --------------------------------------------------------
    # The following test bench comes from the MatLab 'LlrDemap_TestBench2.m'
    # Digital Signal Processing in Modern Communication Systems (Edition 2): Chapter 5 - Section 6: 
    if(TestSelection['SoftDemappingSimpleQPSK'] == True): 
        BitsPerSymbol  = 2   # QPSK
        TxSymbolReal   = np.arange(-1.2, 1.2, 0.01, dtype = np.float32)
        TxSymbolImag   = TxSymbolReal
        TxSymbols      = np.complex64(TxSymbolReal + 1j*TxSymbolImag)
         
        FreqResponse   = np.complex64(1)
        NoiseVariance  = np.float32(0.1)

        NumberOfIterations = len(TxSymbols)
        LLRBits = np.zeros([NumberOfIterations, BitsPerSymbol], dtype = np.float32)

        for SymbolIndex, Symbol in enumerate(TxSymbols):
            LLRBits[SymbolIndex, :] = Mapper.SoftDemapping(BitsPerSymbol
                                                         , Symbol
                                                         , NoiseVariance 
                                                         , FreqResponse)
        
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(TxSymbolReal, LLRBits[:, 0], c = 'red')
        plt.ylabel('LLR dB')
        plt.xlabel('Real Part of Estimate')
        plt.legend(['bit0'])
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(TxSymbolReal, LLRBits[:, 1], c = 'red')
        plt.ylabel('LLR dB')
        plt.xlabel('Imag Part of Estimate')
        plt.legend(['bit1'])
        plt.grid(True)
         
 

    # --------------------------------------------------------
    # SoftDemappingSimple16QAM
    # --------------------------------------------------------
    # The following test bench comes from the MatLab 'LlrDemap_TestBench2.m'
    # Digital Signal Processing in Modern Communication Systems (Edition 2): Chapter 5 - Section 6: 
    if(TestSelection['SoftDemappingSimple16QAM'] == True): 
        BitsPerSymbol  = 4   # 16QAM
        TxSymbolReal   = np.arange(-1.2, 1.2, 0.01, dtype = np.float32)
        TxSymbolImag   = TxSymbolReal
        TxSymbols      = np.complex64(TxSymbolReal + 1j*TxSymbolImag)
        #QamSymbols     = Mapper.Mapping(CurrentERate, np.array([0, 0, 0, 1, 1, 0], dtype=np.int8))
        FreqResponse   = np.complex64(1)
        NoiseVariance  = np.float32(0.1)

        NumberOfIterations = len(TxSymbols)
        LLRBits = np.zeros([NumberOfIterations, BitsPerSymbol], dtype = np.float32)

        for SymbolIndex, Symbol in enumerate(TxSymbols):
            LLRBits[SymbolIndex, :] = Mapper.SoftDemapping(BitsPerSymbol
                                                         , Symbol
                                                         , NoiseVariance 
                                                         , FreqResponse)
        plt.figure(3)   
        plt.subplot(2, 1, 1)
        plt.plot(TxSymbolReal, LLRBits[:, 0], c = 'red')
        plt.plot(TxSymbolReal, LLRBits[:, 1], c = 'blue')
        plt.ylabel('LLR dB')
        plt.xlabel('Real Part of Estimate')
        plt.legend(['bit0', 'bit1'])
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(TxSymbolReal, LLRBits[:, 2], c = 'red')
        plt.plot(TxSymbolReal, LLRBits[:, 3], c = 'blue')
        plt.ylabel('LLR dB')
        plt.xlabel('Imag Part of Estimate')
        plt.legend(['bit2', 'bit3'])
        plt.grid(True)
  


    # --------------------------------------------------------
    # SoftDemappingSimple64QAM
    # --------------------------------------------------------
    # The following test bench comes from the MatLab 'LlrDemap_TestBench2.m'
    # Digital Signal Processing in Modern Communication Systems (Edition 2): Chapter 5 - Section 6: 
    if(TestSelection['SoftDemappingSimple64QAM'] == True): 
        BitsPerSymbol  = 6   # 64QAM
        TxSymbolReal   = np.arange(-1.2, 1.2, 0.01, dtype = np.float32)
        TxSymbolImag   = TxSymbolReal
        TxSymbols      = np.complex64(TxSymbolReal + 1j*TxSymbolImag)
        #QamSymbols     = Mapper.Mapping(CurrentERate, np.array([0, 0, 0, 1, 1, 0], dtype=np.int8))
        FreqResponse   = np.complex64(1)
        NoiseVariance  = np.float32(0.1)

        NumberOfIterations = len(TxSymbols)
        LLRBits = np.zeros([NumberOfIterations, BitsPerSymbol], dtype = np.float32)

        for SymbolIndex, Symbol in enumerate(TxSymbols):
            LLRBits[SymbolIndex, :] = Mapper.SoftDemapping(BitsPerSymbol
                                                         , Symbol
                                                         , NoiseVariance 
                                                         , FreqResponse)
        plt.figure(4)
        plt.subplot(2, 1, 1)
        plt.plot(TxSymbolReal, LLRBits[:, 0], c = 'red')
        plt.plot(TxSymbolReal, LLRBits[:, 1], c = 'blue')
        plt.plot(TxSymbolReal, LLRBits[:, 2], c = 'green')
        plt.ylabel('LLR dB')
        plt.xlabel('Real Part of Estimate')
        plt.legend(['bit0', 'bit1', 'bit2'])
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(TxSymbolReal, LLRBits[:, 3], c = 'red')
        plt.plot(TxSymbolReal, LLRBits[:, 4], c = 'blue')
        plt.plot(TxSymbolReal, LLRBits[:, 5], c = 'green')
        plt.ylabel('LLR dB')
        plt.xlabel('Imag Part of Estimate')
        plt.legend(['bit3', 'bit4', 'bit5'])
        plt.grid(True)
        
    plt.show()
 

    # --------------------------------------------------------
    # BER Testing for all constellations
    # --------------------------------------------------------
    # The following test bench determines the bit error rate of the different constellations
    # as TX constellations are noise infested and hard demapped    
    if(TestSelection['BerTest'] == True): 
        # Generate Random Bits
        r           = np.random.RandomState()
        CinrdBRange = list(range(-5, 30)) 
        BerBpsk     = [0] * len(CinrdBRange)
        BerQpsk     = [0] * len(CinrdBRange)
        Ber16Qam    = [0] * len(CinrdBRange)
        Ber64Qam    = [0] * len(CinrdBRange)
        for Index, CinrdB  in enumerate(CinrdBRange):
            print(str(CinrdB))
            InputBits   = np.random.randint(low=0, high=2, size = 60000, dtype = np.uint8)
            Qam1Symbols = Mapper.Mapping(1, InputBits)
            Qam2Symbols = Mapper.Mapping(2, InputBits)
            Qam4Symbols = Mapper.Mapping(4, InputBits)
            Qam6Symbols = Mapper.Mapping(6, InputBits)

            # Add additive white Gaussian noise
            Qam1Noisy = AddAwgn(CinrdB, Qam1Symbols)
            Qam2Noisy = AddAwgn(CinrdB, Qam2Symbols)
            Qam4Noisy = AddAwgn(CinrdB, Qam4Symbols)
            Qam6Noisy = AddAwgn(CinrdB, Qam6Symbols)

            # Hard Demapping
            Qam1Output = Mapper.HardDemapping(1, Qam1Noisy)
            Qam2Output = Mapper.HardDemapping(2, Qam2Noisy)
            Qam4Output = Mapper.HardDemapping(4, Qam4Noisy)
            Qam6Output = Mapper.HardDemapping(6, Qam6Noisy)

            # Count Errors and compute BER
            Qam1Errors = np.sum((Qam1Output + InputBits) % 2)
            Qam2Errors = np.sum((Qam2Output + InputBits) % 2)
            Qam4Errors = np.sum((Qam4Output + InputBits) % 2)
            Qam6Errors = np.sum((Qam6Output + InputBits) % 2)

            BerBpsk[Index]  = Qam1Errors / len(InputBits)
            BerQpsk[Index]  = Qam2Errors / len(InputBits)
            Ber16Qam[Index] = Qam4Errors / len(InputBits)
            Ber64Qam[Index] = Qam6Errors / len(InputBits)


        # Let's create a curve fit to the data to get some better behaved curves
        # BPSK
        InBpsk  = []; InQpsk  = []; In16Qam  = []; In64Qam  = []
        OutBpsk = []; OutQpsk = []; Out16Qam = []; Out64Qam = []
        for Index in range(0, len(CinrdBRange)):
            if BerBpsk[Index] > 0.0001:
                InBpsk.append(CinrdBRange[Index])
                OutBpsk.append(10*np.log10(BerBpsk[Index]))
            if BerQpsk[Index] > 0.0001:
                InQpsk.append(CinrdBRange[Index])
                OutQpsk.append(10*np.log10(BerQpsk[Index]))
            if Ber16Qam[Index] > 0.0001:
                In16Qam.append(CinrdBRange[Index])
                Out16Qam.append(10*np.log10(Ber16Qam[Index]))
            if Ber64Qam[Index] > 0.0001:
                In64Qam.append(CinrdBRange[Index])
                Out64Qam.append(10*np.log10(Ber64Qam[Index]))

        DegsFreedom = 5   # Degrees of freedom (Number of coefficients)
        # BPSK
        F   = np.zeros([len(InBpsk), DegsFreedom], np.float32)
        for row in range(0, len(InBpsk)):
            for column in range(0, DegsFreedom):
                F[row, column] = InBpsk[row]**column

        ABpsk = (np.linalg.inv(F.transpose().dot(F))).dot(F.transpose().dot(OutBpsk))

        CinrRangeBpsk = np.arange(-2, 10, 1, np.float32)
        BerRangeBpsk  = np.zeros(len(CinrRangeBpsk), np.float32)
        for Index in range(0, DegsFreedom):
            BerRangeBpsk += ABpsk[Index]*(CinrRangeBpsk**Index)    

       
        # QPSK
        F   = np.zeros([len(InQpsk), DegsFreedom], np.float32)
        for row in range(0, len(InQpsk)):
            for column in range(0, DegsFreedom):
                F[row, column] = InQpsk[row]**column

        AQpsk = (np.linalg.inv(F.transpose().dot(F))).dot(F.transpose().dot(OutQpsk))

        CinrRangeQpsk = np.arange(-2, 15, 1, np.float32)
        BerRangeQpsk  = np.zeros(len(CinrRangeQpsk), np.float32)
        for Index in range(0, DegsFreedom):
            BerRangeQpsk += AQpsk[Index]*(CinrRangeQpsk**Index)  

        # 16Qam
        F   = np.zeros([len(In16Qam), DegsFreedom], np.float32)
        for row in range(0, len(In16Qam)):
            for column in range(0, DegsFreedom):
                F[row, column] = In16Qam[row]**column

        A16Qam = (np.linalg.inv(F.transpose().dot(F))).dot(F.transpose().dot(Out16Qam))

        CinrRange16Qam = np.arange(-2, 23, 1, np.float32)
        BerRange16Qam  = np.zeros(len(CinrRange16Qam), np.float32)
        for Index in range(0, DegsFreedom):
            BerRange16Qam += A16Qam[Index]*(CinrRange16Qam**Index) 


        # 64Qam
        F   = np.zeros([len(In64Qam), DegsFreedom], np.float32)
        for row in range(0, len(In64Qam)):
            for column in range(0, DegsFreedom):
                F[row, column] = In64Qam[row]**column

        A64Qam = (np.linalg.inv(F.transpose().dot(F))).dot(F.transpose().dot(Out64Qam))

        CinrRange64Qam = np.arange(-2, 29, 1, np.float32)
        BerRange64Qam  = np.zeros(len(CinrRange64Qam), np.float32)
        for Index in range(0, DegsFreedom):
            BerRange64Qam += A64Qam[Index]*(CinrRange64Qam**Index) 


        print('ABpsk: ' + str(ABpsk))
        print('AQpsk: ' + str(AQpsk))
        print('A16Qam: ' + str(A16Qam))
        print('A64Qam: ' + str(A64Qam))

        plt.figure(5)
        plt.semilogy(CinrdBRange, BerBpsk, c = 'red')
        plt.semilogy(CinrRangeBpsk, 10**(BerRangeBpsk/10), 'r:')
        plt.semilogy(CinrdBRange, BerQpsk, c = 'blue')
        plt.semilogy(CinrRangeQpsk, 10**(BerRangeQpsk/10), 'b:')
        plt.semilogy(CinrdBRange, Ber16Qam, c = 'green')
        plt.semilogy(CinrRange16Qam, 10**(BerRange16Qam/10), 'g:')
        plt.semilogy(CinrdBRange, Ber64Qam, c = 'black')
        plt.semilogy(CinrRange64Qam, 10**(BerRange64Qam/10), 'k:')
        #plt.legend(['BPSK', 'QPSK', '16QAM', '64QAM'])
        plt.grid(True)
        plt.title('BER Curves')
        plt.ylabel('Probability')
        plt.xlabel('SNR in dB')
        plt.show()

        stop = 1

    