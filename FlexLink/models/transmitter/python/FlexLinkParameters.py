# File:  FlexLinkParemeters.py
# Notes: This files provides basic parameters for the FlexLink Wireless Communication standard.

__title__     = "FlexLinkParemeters"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Sept, 4rd, 2022"
__copyright__ = 'Andreas Schwarzinger'

from   enum               import unique, Enum
from   FlexLinkCoder      import CCrcProcessor 
import math
import numpy              as     np
import matplotlib.pyplot  as     plt
from   matplotlib.colors  import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches



# -----------------------------------------------------------------
# > Declare and enumerate the possible Bandwidths
# -----------------------------------------------------------------
@unique
class EBandwidth(Enum):
    """
    This class enumerates the different bandwidth options
    """
    Bw5MHz      = 5
    Bw10MHz     = 10
    Bw20MHz     = 20
    Bw40MHz     = 40

    # This function verifies that we currently support this listed bandwidth
    @classmethod
    def CheckValidOption(cls, EnumInput) -> bool:
        assert isinstance(EnumInput, EBandwidth), 'The EnumInput is of improper type'
        UnsupportedBwList   = [5, 10, 40]  
        bUnsupportedBw      = any([x == EnumInput.value for x in UnsupportedBwList])
        assert bUnsupportedBw == False,    'An unsupported bandwidth was requested.'




# -----------------------------------------------------------------
# > Declare and enumerate the possible subcarrier spacings
# -----------------------------------------------------------------
@unique
class ESubcarrierSpacing(Enum):
    """
    This class enumerates the different subcarrier spacing options
    """
    Sc20KHz     = 20000
    Sc40KHz     = 40000
    Sc80KHz     = 80000
    Sc160KHz    = 160000

    # This function verifies that we currently support the requested subcarrier spacing
    @classmethod
    def CheckValidOption(cls, EnumInput) -> bool:
        assert isinstance(EnumInput, ESubcarrierSpacing), 'The EnumInput is of improper type'
        UnsupportedScList   = [40000, 80000, 160000]  
        bUnsupportedSc      = any([x == EnumInput.value for x in UnsupportedScList])
        assert bUnsupportedSc == False,    'An unsupported subcarrier spacing was requested.'





# ----------------------------------------------------------------
# > Declare and enumerate the possible Cyclic Prefix Options
# ----------------------------------------------------------------
@unique
class ECyclicPrefix(Enum):
    """
    This class enumerates the different cyclic prefix options
    """
    Ec8MicroSec  = 8e-6
    Ec4MicroSec  = 4e-6
    Ec2MicroSec  = 2e-6
    Ex1MicroSec  = 1e-6

    # This function verifies that we currently support this listed bandwidth
    @classmethod
    def CheckValidOption(cls, EnumInput) -> bool:
        assert isinstance(EnumInput, ECyclicPrefix), 'The EnumInput is of improper type'
        UnsupportedCpList   = [1e-6, 2e-6, 8e-6]  
        bUnsupportedCp      = any([x == EnumInput.value for x in UnsupportedCpList])
        assert bUnsupportedCp == False,    'An unsupported cyclic prefix was requested.'






# ---------------------------------------------------------------
# > Declare and enumerate all possible resource elements types
# ---------------------------------------------------------------
@unique
class EReType(Enum):
    Unknown          = -1    # Not assigned
    DmrsPort0        = 0     # Demodulation reference signals port 0
    DmrsPort1        = 1     # Demodulation reference signals port 1
    PhaseRs          = 2     # Phase reference signals port 0
    Control          = 4     # Control resource element
    DataSignalField  = 5     # Data resource element (Signal Field)
    DataPayloadA     = 6     # Data for PayloadA
    DataPayloadB     = 7     # Data for PayloadB
    Emtpy            = 8     # Empty resource element = 0 + j0
    
    # This function verifies that we currently support the requested subcarrier spacing
    @classmethod
    def CheckValidOption(cls, EnumInput) -> bool:
        assert isinstance(EnumInput, EReType), 'The EnumInput is of improper type.'
        





# ----------------------------------------------------------------
# > Define a resource elements
# ----------------------------------------------------------------
class CResourceElement():
    '''
    A CResourceElement is defined by its value, frequency, time and type
    '''

    # ---------------------------------------------------------
    # Constructor
    def __init__(self
               , FreqUnit:  int   = 0
               , TimeUnit:  int   = 0
               , Value:     complex = 0 + 0j
               , Type:      EReType = EReType.Unknown
               , Sc:        ESubcarrierSpacing = ESubcarrierSpacing.Sc20KHz
               , Cp:        ECyclicPrefix      = ECyclicPrefix.Ec4MicroSec):

        self.FreqUnit           = FreqUnit
        self.TimeUnit           = TimeUnit
        self.FrequencyHz        = FreqUnit * Sc.value
        self.TimeSec            = TimeUnit * ((1/Sc.value) + Cp.value)
        self.TxValue            = Value
        self.RxValue            = Value
        self.Type               = Type
        self.Sc                 = Sc   # Subcarrier spacing
        self.Cp                 = Cp   # Cyclic prefix
        self.IdealFreqResponse  = 0    # The ideal/actual frequency response computed by the model
        self.RawFreqResponse    = 0    # The measured raw frequency response
        self.EstFreqResponse    = 0    # The frequency response estimate after noise reduction
        self.EqualizeRxValue    = 0    # The equalized value 

    # -------------------------------------------------------
    # Reconstructor
    def __call__(self
               , FreqUnit:  int   = 0
               , TimeUnit:  int   = 0
               , Value:     complex = 0 + 0j
               , Type:      EReType = EReType.Unknown
               , Sc:        ESubcarrierSpacing = ESubcarrierSpacing.Sc20KHz
               , Cp:        ECyclicPrefix      = ECyclicPrefix.Ec4MicroSec):

        self.FreqUnit           = FreqUnit
        self.TimeUnit           = TimeUnit
        self.FrequencyHz        = FreqUnit * Sc.value
        self.TimeSec            = TimeUnit * ((1/Sc.value) + Cp.value)
        self.TxValue            = Value
        self.RxValue            = Value
        self.Type               = Type
        self.Sc                 = Sc   # Subcarrier spacing
        self.Cp                 = Cp   # Cyclic prefix
        self.IdealFreqResponse  = 0    # The ideal/actual frequency response computed by the model
        self.RawFreqResponse    = 0    # The measured raw frequency response
        self.EstFreqResponse    = 0    # The frequency response estimate after noise reduction
        self.EqualizeRxValue    = 0    # The equalized value 

    # ---------------------------------------------------------
    @staticmethod
    def CreateReArray(FreqUnitArray:       list 
                    , TimeUnitArray:       list 
                    , Type:                EReType            = EReType.Unknown
                    , Sc:                  ESubcarrierSpacing = ESubcarrierSpacing.Sc20KHz
                    , Cp:                  ECyclicPrefix      = ECyclicPrefix.Ec4MicroSec):

        """
        brief: CreateReArray() creates a array of resource elements in a rectangular fashion.
        param: FreqUnitArray       - An array in frequency units indicating the subcarriers tone index
        param: TimeUnitArray       - An array in time units indicating the OFDM symbol index
        param: Type                - The type of resource element.
        param: Sc                  - The subcarrier spacing Enum
        param: Cp                  - The cyclic prefix Enum
        """

        # -----------------------------------------------------
        # Check Errors
        assert isinstance(FreqUnitArray, list)
        assert isinstance(TimeUnitArray, list)
        EReType.CheckValidOption(Type)
        ESubcarrierSpacing.CheckValidOption(Sc)
        ECyclicPrefix.CheckValidOption(Cp)

        # -----------------------------------------------------
        # Create The list
        Array           = []
        for TimeUnit in TimeUnitArray:
            for FreqUnit in FreqUnitArray:
                Array.append(CResourceElement(FreqUnit
                                            , TimeUnit
                                            , 0 + 0j
                                            , Type
                                            , Sc
                                            , Cp))
        return Array







# ---------------------------------------------------------------
# > Declare and define the control information
# ---------------------------------------------------------------
class CControlInformation():
    """
    brief:   The class manages the control information embedded in the first pilot/reference symbol
    """
    Boolean                                  = [False, True]
    AvailableTxAntennaPorts                  = [1, 2]         # Number of Tx Antenna ports for which Reference Signals exist
    AvailableSignalFieldSymbols              = [1, 2, 3, 4]   # The index into this array is Ni
    AvailableNumberOfDcCarriers              = [1, 3]         # The index into this array is Dci
    AvailableReferenceSymbolPeriodicity20Khz = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
                                                              # The OFDM symbol periodicity for subcarrier spacing of 20KHz
                                                              # For the other subcarrier spacings we need to multiply these
                                                              # values by 2 ** (SubcarrierSpacing/20KHz)
                                                              # 40KHz ->  [ 4,   8, ...,   64]
                                                              # 80KHz ->  [ 8,  16, ...,  128]
                                                              # 160Khz -> [ 16, 32, ...,  256] 

    NumberOfControlBits                       =  4 + 2 + 1 + 1 + 1 + 1 + 1 + 3  # = 14 bits -> Pi / Ni / Ai / Phi / Bi / Qi / Dci / User


    # -----------------------------------------------------------
    # >> The constructor
    # -----------------------------------------------------------   
    def __init__(self
               , ESc:  ESubcarrierSpacing = ESubcarrierSpacing.Sc20KHz   # -> ESc = The subcarrier spacing (Needed for final Ref Symbol periodicity)
               , Pi:   int = 0   # Pi  = 0/1.. /15  -> Pi  = Index for the period in OFDM symbols between reference sybmols. 
               , Ni:   int = 0   # Ni  = 0/1/2/3    -> Ni  = Index for number of Ofdm symbols in the signal field
               , Ai:   int = 0   # Ai  = 0/1        -> Ai  = Index for number of Tx Antenna ports for which reference signals are available.                      
               , Phi:  int = 0   # Phi = 0/1        -> Phi = No / Yes Phase reference signals  
               , Bi:   int = 0   # Ri  = 0/1        -> Bi  = No / Yes Reference signals boosting by sqrt(2) in magnitude
               , Qi:   int = 0   # Qi  = 0/1        -> Qi  = BPSK / QPSK
               , Dci:  int = 0   # Dci = 0/1        -> Dci = 0 /1  indicates 1 Center DC subcarrier or 3 Center DC subcarriers
               , User: int = 0): # User = 0/1.../7  -> These are just user selectable bits      


        # Basic error checking
        assert isinstance(ESc,  ESubcarrierSpacing) , 'The ESc  input argument must be of type ESubcarrierSpacing'
        assert isinstance(Pi,   int),                 'The Pi   input argument is not an integer'
        assert isinstance(Ni,   int),                 'The Ni   input argument is not an integer'
        assert isinstance(Ai,   int),                 'The Ai   input argument is not an integer'
        assert isinstance(Phi,  int),                 'The Phi  input argument is not an integer'
        assert isinstance(Bi,   int),                 'The Bi   input argument is not an integer'
        assert isinstance(Qi,   int),                 'The Qi   input argument is not an integer' 
        assert isinstance(Dci,  int),                 'The Dci  input argument is not an integer' 
        assert isinstance(User, int),                 'The User input argument is not an integer'

        assert Pi   >= 0 and Pi   < 16,                                                   'The Pi  input argument is out of range'
        assert Ni   >= 0 and Ni   < len(CControlInformation.AvailableSignalFieldSymbols), 'the Ni  input argument is out of range' 
        assert Ai   >= 0 and Ai   < len(CControlInformation.AvailableTxAntennaPorts),     'The Ai  input argument is out of range'
        assert Phi  >= 0 and Phi  < 2,                                                    'The Phi input argument is out of range'
        assert Bi   >= 0 and Bi   < 2,                                                    'The Bi  input argument is out of range'
        assert Qi   >= 0 and Qi   < 2,                                                    'The Qi  input argument is out of range'
        assert Dci  >= 0 and Dci  < len(CControlInformation.AvailableNumberOfDcCarriers), 'The Dci input argument is out of range'       
        assert User >= 0 and User < 8,                                                    'The User input is out of range'

        # Save Input Arguments
        self.ESc  = ESc
        self.Pi   = Pi
        self.Ni   = Ni
        self.Ai   = Ai
        self.Phi  = Phi
        self.Bi   = Bi
        self.Qi   = Qi
        self.Dci  = Dci
        self.User = User

        # Process the input arguments
        self.NumberOfTxAntennaPorts         = CControlInformation.AvailableTxAntennaPorts[self.Ai]
        self.NumberOfSignalFieldSymbols     = CControlInformation.AvailableSignalFieldSymbols[self.Ni]
        Multiplier = 2 ** (int(self.ESc.value / 20000) - 1)  # Check out the explanation above
        self.ReferenceSymbolPeriodicity     = CControlInformation.AvailableReferenceSymbolPeriodicity20Khz[self.Pi] * Multiplier
        self.UsingPhaseRefSignals           = CControlInformation.Boolean[self.Phi]
        self.ReferenceSignalBoosting        = CControlInformation.Boolean[self.Bi]
        self.SignalFieldIsQpsk              = CControlInformation.Boolean[self.Qi]
        self.NumberDcCarriers               = CControlInformation.AvailableNumberOfDcCarriers[self.Dci]

        # Build the Control Bit array of 14 bits --> 4 + 2 + 1 + 1 + 1 + 1 + 1 + 3
        self.ControlBitString = (bin(Pi)[2:]).zfill(4) + \
                                (bin(Ni)[2:]).zfill(2) + \
                                (bin(Ai)[2:]).zfill(1) + \
                                (bin(Phi)[2:]).zfill(1) + \
                                (bin(Bi)[2:]).zfill(1) + \
                                (bin(Qi)[2:]).zfill(1) + \
                                (bin(Dci)[2:]).zfill(1) + \
                                (bin(User)[2:]).zfill(3) 
        
        # When we map the control bit vector, we repeat each bit a certain number of times.
        self.ControlBitVector = [int(a) for a in self.ControlBitString]


    # ------------------------------------------------------------
    # >> Overload the str() function
    # ------------------------------------------------------------
    def __str__(self):
        ReturnString  = '------------- Summary of Control Information ----------------- \n'
        ReturnString += 'Reference symbols appear every:                 ' + str(self.ReferenceSymbolPeriodicity) + " Ofdm symbols.\n"  
        ReturnString += 'Number of Signal Field Ofdm symbols:            ' + str(self.NumberOfSignalFieldSymbols) + ".\n" 
        ReturnString += 'Number of Tx Antennas:                          ' + str(self.NumberOfTxAntennaPorts) + ".\n"
        ReturnString += 'Phase reference signals enabled:                ' + str(self.UsingPhaseRefSignals) + '.\n'
        ReturnString += 'Reference signals (pilots) boosting by sqrt(2): ' + str(self.ReferenceSignalBoosting) + '\n'
        ReturnString += 'Is the signal field QPSK (true) / BPSK (false): ' + str(self.SignalFieldIsQpsk) + '\n'
        ReturnString += 'The number of DC subcarriers is:                ' + str(self.NumberDcCarriers) + '\n'
        ReturnString += 'The three user defined bits have value:         ' + str(self.User)  + '\n'
        ReturnString += 'The control bit vector is:                      ' + self.ControlBitString
        return ReturnString









# ---------------------------------------------------------------
# > Declare and define the signal field information
# ---------------------------------------------------------------
class CSignalField():
    """
    The class manages the information embedded in the signal field
    """
    NumberCrcBits                = 16
    NumberOfUserBits             = 24
 
    AvailableEncodedBlockSizes   = [648, 1296, 1944, 0]
    AvailableFecOptions          = ['LDPC_1_2', 'LDPC_2_3', 'LDPC_3_4', 'LDPC_3_4']    # Identical to WLAN 802.11n (See annex R)
    BitMultiplier                = [2, 3/2, 4/3, 0]                                    # LDPC_5_6 not yet supported in FlexLink
    RateMatchingMultiplier       = [1, 1.5, 2, 3, 4, 6, 8, 16]       # Repetition percentage
                                                                     # 0 -> 1    - No bits are repeated
                                                                     # 1 -> 1.5  - Repeat every second bit once
                                                                     # 2 -> 2    - Repeat every bit once
                                                                     # 3 -> 3    - Repeat every bit twice
                                                                     # 4 -> 4    - Repeat every bit three times
                                                                     # 5 -> 6    - Repeat every bit five times
                                                                     # 7 -> 8    - Repeat every bit seven times
                                                                     # 8 -> 16   - Repeat every bit fifteen times
    AvailableBitsPerSymbol       = [1, 2, 4, 6]       # BPSK, QPSK, 16QAM, 64QAM

    # Number of bits:
    # TBSi (2bits), NTBS (16bits), FECi (2bits), RMi (2bits), BPSi (2bits), UserBits (24), Crc (16) = 88 bits
    # LDPC 1/2     => 88  * 2  = 176
    # RateMatching => rate match until all subcarriers in the number of available OFDM symbols are filled
    NumberDataBitsSignalField    = 2*(2 + 16 + 3 + 2 + 2) + NumberOfUserBits + NumberCrcBits  #  Data 90 bits               
    NumberEncodedBitsSignalField = NumberDataBitsSignalField * 2                              #  Encode 180 bits                                                           


    # -----------------------------------------------------------
    # >> The constructor
    # -----------------------------------------------------------   
    def __init__(self
               , EBS1i: int = 0       # The encoded block size index   for Payload 1
               , NTBS1: int = 0       # The number of transport blocks for Payload 1
               , FEC1i: int = 0       # The forward error correction   for Payload 1
               , RM1i:  int = 0       # The rate matching index        for Payload 1
               , BPS1i: int = 0       # The bits/symbol index          for Payload 1
               , EBS2i: int = 0       # The encoded block size index   for Payload 2
               , NTBS2: int = 0       # The number of transport blocks for Payload 2
               , FEC2i: int = 0       # The forward error correction   for Payload 2
               , RM2i:  int = 0       # The rate matching index        for Payload 2
               , BPS2i: int = 0       # The bits/sybmol index          for Payload 2
               , UserBits: list[int] = [0] * 24):

        # Basic type checking
        assert isinstance(EBS1i, int) and isinstance(EBS2i,  int), 'The EBS index input arguments are not integers.'
        assert isinstance(NTBS1, int) and isinstance(NTBS2,  int), 'The NTBS input arguments are not integers.'
        assert isinstance(FEC1i, int) and isinstance(FEC2i,  int), 'The FEC index input arguments are not integers.'
        assert isinstance(RM1i,  int) and isinstance(RM2i,   int), 'The rate matching index input arguments are not integers.'
        assert isinstance(BPS1i, int) and isinstance(BPS2i,  int), 'The rate matching index input arguments are not integers.'
        assert isinstance(UserBits, list),                         'The user bit array is not a list' 
        assert len(UserBits) == 24,                                'The number of user bits must be 24.'
        bProperBits = all([( (x == 1 or x == 0) and isinstance(x, int)) for x in UserBits])
        assert bProperBits,                                       'The user bits must be integers of value 0 or 1.'

        # Range checking
        assert EBS1i >= 0 and EBS1i < 3,     'The EBS1i input argument is out of range.'
        assert NTBS1 >= 0 and NTBS1 < 65536, 'The NTBS1 input argument is out of range.'
        assert FEC1i >= 0 and FEC1i < 3,     'The FEC1i input argument is out of range.'
        assert RM1i  >= 0 and RM1i  < 8,     'The RM1i  input argument is out of range.'
        assert BPS1i >= 0 and BPS1i < 4,     'The BPS1i input argument is out of range.'

        assert EBS2i >= 0 and EBS2i < 3,     'The EBS2i input argument is out of range.'
        assert NTBS2 >= 0 and NTBS2 < 65536, 'The NTBS2 input argument is out of range.'
        assert FEC2i >= 0 and FEC2i < 3,     'The FEC2i input argument is out of range.'
        assert RM2i  >= 0 and RM2i  < 8,     'The RM2i  input argument is out of range.'
        assert BPS2i >= 0 and BPS2i < 4,     'The BPS1i input argument is out of range.'

        # Member variables
        self.EBS1     = CSignalField.AvailableEncodedBlockSizes[EBS1i]
        assert  np.fmod(self.EBS1 / CSignalField.BitMultiplier[FEC1i], 1) == 0
        self.TBS1     = math.floor(self.EBS1 / CSignalField.BitMultiplier[FEC1i])
        self.NTB1     = NTBS1
        self.FEC1     = CSignalField.AvailableFecOptions[FEC1i]
        self.RM1      = CSignalField.RateMatchingMultiplier[RM1i]
        self.BPS1     = CSignalField.AvailableBitsPerSymbol[BPS1i]

        self.EBS2     = CSignalField.AvailableEncodedBlockSizes[EBS2i]
        assert  np.fmod(self.EBS2 / CSignalField.BitMultiplier[FEC2i], 1) == 0
        self.TBS2     = math.floor(self.EBS2 / CSignalField.BitMultiplier[FEC2i])
        self.NTB2     = NTBS2
        self.FEC2     = CSignalField.AvailableFecOptions[FEC2i]
        self.RM2      = CSignalField.RateMatchingMultiplier[RM2i]
        self.BPS2     = CSignalField.AvailableBitsPerSymbol[BPS2i]
        self.UserBits = UserBits

        self.NumberOfMacDataBits1   = (self.TBS1 - CSignalField.NumberCrcBits) * self.NTB1
        self.NumberOfTransportBits1 = self.TBS1 * self.NTB1
        self.NumberEncodedBits1     = self.NumberOfTransportBits1 * CSignalField.BitMultiplier[FEC1i] 
        self.NumberRateMatchedBits1 = self.NumberEncodedBits1 * self.RM1
        self.NumberQamSymbols1      = int(math.ceil(self.NumberRateMatchedBits1 / self.BPS1))
        self.NumberOfMacDataBits2   = (self.TBS2 -  CSignalField.NumberCrcBits) * self.NTB2
        self.NumberOfTransportBits2 = self.TBS2 * self.NTB2
        self.NumberEncodedBits2     = self.NumberOfTransportBits2 * CSignalField.BitMultiplier[FEC2i] 
        self.NumberRateMatchedBits2 = self.NumberEncodedBits2 * self.RM2    
        self.NumberQamSymbols2      = int(math.ceil(self.NumberRateMatchedBits1 / self.BPS2))

        # Build the bit string 
        self.BitString  = ''
        self.BitString += "{:02b}".format(EBS1i)  
        self.BitString += "{:016b}".format(NTBS1)  
        self.BitString += "{:02b}".format(FEC1i)  
        self.BitString += "{:03b}".format(RM1i)  
        self.BitString += "{:02b}".format(BPS1i)  
        self.BitString += "{:02b}".format(EBS2i)  
        self.BitString += "{:016b}".format(NTBS2)  
        self.BitString += "{:02b}".format(FEC2i)  
        self.BitString += "{:03b}".format(RM2i)  
        self.BitString += "{:02b}".format(BPS2i)  
        for Index in range(0, 24):
            self.BitString += "{:01b}".format(self.UserBits[Index])

        # Build the CRC for the bit string 
        self.BitVector   = [int(Bit) for Bit in self.BitString]
        CrcOutput        = CCrcProcessor.ComputeCrc(16, self.BitVector)
        for Index in range(0, len(CrcOutput)):
            self.BitVector.append(int(CrcOutput[Index]))
            self.BitString += str(CrcOutput[Index])




    # ------------------------------------------------------------
    # >> Overload the str() function
    # ------------------------------------------------------------
    def __str__(self):
        ReturnString  = '------------- Summary of Signal Field ----------------- \n'
        ReturnString += 'NumberDataBitsSignalField    = 2 (for 2 payloads) * (2 (for EBS) + 16 (for NTBS) + 2 (for FEC) + 3 (for RM) + 2 (for BPS)) + 24 (for user data) + 16 (for crc) =  Data 90 bits' + '\n'               
        ReturnString += 'NumberEncodedBitsSignalField = 180 (as we use 1/2 rate encoding)'    + '\n'     
        ReturnString += '-> For payload 1\n'    
        ReturnString += 'TBS1:     ' + str(self.TBS1)  + '   bits per transport block\n'
        ReturnString += 'EBS1:     ' + str(self.EBS1)  + '   bits per encoded block\n'
        ReturnString += 'NTB1:     ' + str(self.NTB1)  + '   transport blocks.\n'
        ReturnString += 'FEC1:     ' + str(self.FEC1)  + '\n'
        ReturnString += 'RM1 :     ' + str(self.RM1)   + '   Rate matching multiplier\n'
        ReturnString += 'BPS1:     ' + str(self.BPS1)  + '   bits per QAM symbol\n'
        ReturnString += 'Number of MAC bits         : ' + str(self.NumberOfMacDataBits1) + '\n'
        ReturnString += 'Number of transport bits   : ' + str(self.NumberOfTransportBits1) + '\n'
        ReturnString += 'Number of encoded bits     : ' + str(self.NumberEncodedBits1) + '\n'
        ReturnString += 'Number of rate matched bits: ' + str(self.NumberRateMatchedBits1) + '\n'
        ReturnString += 'Number of QAM symbols:       ' + str(self.NumberQamSymbols1) + '\n'
        ReturnString += '-> For payload 2\n'
        ReturnString += 'TBS2:     ' + str(self.TBS2)  + '   bits per transport block\n'
        ReturnString += 'EBS2:     ' + str(self.EBS2)  + '   bits per encoded block\n'
        ReturnString += 'NTB2:     ' + str(self.NTB2)  + '   transport blocks.\n'
        ReturnString += 'FEC2:     ' + str(self.FEC2)  + ' \n'
        ReturnString += 'RM2 :     ' + str(self.RM2)   + '   Rate matching multiplier\n'
        ReturnString += 'BPS2:     ' + str(self.BPS2)  + '   bits per QAM symbol\n'
        ReturnString += 'Number of MAC bits         : ' + str(self.NumberOfMacDataBits2) + '\n'
        ReturnString += 'Number of transport bits   : ' + str(self.NumberOfTransportBits2)  + '\n'
        ReturnString += 'Number of encoded bits     : ' + str(self.NumberEncodedBits2) + '\n'
        ReturnString += 'Number of rate matched bits: ' + str(self.NumberRateMatchedBits2) + '\n'
        ReturnString += 'Number of QAM symbols:       ' + str(self.NumberQamSymbols2) + '\n'
        ReturnString += 'UserBits: ' + str(self.UserBits) + '\n\n'
        ReturnString += 'Total bit string including attached CRC: ' + self.BitString
        return ReturnString       









 # --------------------------------------------------------------
 # > Declare and define the FlexLinkConfig object
 # --------------------------------------------------------------
class CFlexLinkConfig():
    """
    This class embodies the configuration of the FlexLink physical layer
    """
    ReservedResourceElement          = 0
    ControlResourceElement           = 1
    DmrsReferenceResourceElementP0   = 2
    DmrsReferenceResourceElementP1   = 3
    PhaseReferenceResourceElementP0  = 4
    PhaseReferenceResourceElementP1  = 5
    DataResourceElementSignalField   = 6
    DataResourceElementPayloadA      = 7
    DataResourceElementPayloadB      = 8

    # -----------------------------------------------------------
    # >> The constructor
    # -----------------------------------------------------------
    def __init__(self
               , EBw: EBandwidth          # This information cannot be inferred by the receiver and must be provided by MAC
               , ESc: ESubcarrierSpacing  # This information cannot be inferred by the receiver and must be provided by MAC
               , ECp: ECyclicPrefix       # This information cannot be inferred by the receiver and must be provided by MAC
               , ControlInfo: CControlInformation
               , SignalField: CSignalField
               , BaseSampleRate:  int     # The base sample rate should be 20MHz (supported for now), 
                                          # 20.48MHz or 19.2MHz (unsupported for now as it yields strange a strange number of CP samples)
               , OSR: int = 2):           # The over sample rate (should be 1, 2, or 4) 

        # -----------------------------------
        # Type and Error checking
        # -----------------------------------
        EBandwidth.CheckValidOption(EBw)
        ESubcarrierSpacing.CheckValidOption(ESc)
        ECyclicPrefix.CheckValidOption(ECp)
        assert isinstance(ControlInfo, CControlInformation)
        assert isinstance(SignalField, CSignalField)
        assert isinstance(BaseSampleRate, int) or isinstance(BaseSampleRate, float)
        assert isinstance(OSR, int) 

        assert EBw == EBandwidth.Bw20MHz,         'Currently only the 20MHz bandwidth is supported'
        assert ESc == ESubcarrierSpacing.Sc20KHz, 'Currently only the 20KHz subcarrier spacing is supported' 
        assert ECp == ECyclicPrefix.Ec4MicroSec,  'Currently only the 4 usec cyclic prefix is supported'
        assert ESc == ControlInfo.ESc,            'The subcarrier spacings do not agree'
        # assert BaseSampleRate == 20e6,            'For now we chose to only support 20e6 MHz as the base sample rate'
        assert OSR == 1 or OSR == 2 or OSR == 4,  'The provided OSR is not valid'

        # Save inputs
        self.EBw = EBw
        self.ESc = ESc
        self.ECp = ECp
        self.ControlInfo    = ControlInfo
        self.SignalField    = SignalField
        self.BaseSampleRate = BaseSampleRate 
        self.OSR            = OSR

        # Copy to member attributes
        self.Bw                   = EBw.value * 1000000   # Hz      (int)
        self.CpDurationSec        = ECp.value             # Seconds (float)
        self.CpDurationSamples    = math.floor(self.CpDurationSec * self.BaseSampleRate * self.OSR)

        self.NumTriplets = 0
        self.FftSize     = 0
        if EBw == EBandwidth.Bw20MHz:
            match ESc.value:
                case 20000:
                    self.NumTriplets = 300
                    self.FftSize     = 1024 * self.OSR
                case _:
                    assert False, 'Improper subcarrier spacing.'

        self.SubcarrierSpacing  = self.OSR * self.BaseSampleRate / self.FftSize
        assert abs(self.SubcarrierSpacing - self.ESc.value) < 1000, \
                 'The FftSize and sample rate are incompatible with the desired subcarrier spacing.'

        # Compute some range parameters
        self.NumSubcarriers       = self.NumTriplets * 3 + 1
        self.NumHalfSubcarriers   = math.floor(self.NumSubcarriers/2)
        
        self.MaxTone              = +self.NumHalfSubcarriers
        self.MinTone              = -self.NumHalfSubcarriers
        self.DoubleSidedBandwidth =  self.NumSubcarriers * ESc.value

        # Array that will hold the layout once computed 
        # Column 1 holds the tone index
        # Column 2 holds the nature of the resource element
        self.LayoutFirstReferenceSymbol  = np.zeros([2, self.NumSubcarriers])
        self.LayoutOtherReferenceSymbols = np.zeros([2, self.NumSubcarriers])
        self.LayoutDataSymbols           = np.zeros([2, self.NumSubcarriers])

        # Compute the layout of each symbol type
        #self.ComputeSymbolLayout()
                

    # --------------------------------------------------------------
    # > Compute Resource Element Layout in the Resource Grid
    # --------------------------------------------------------------
    def ComposeSymbolLayout(self):
        '''
        :brief: This function determines the locatino of reference, control and data resourc elements in the resource grid
        '''
        



    # --------------------------------------------------------------
    # > Compute Layout of each OFDM symbol
    # --------------------------------------------------------------
    def ComputeSymbolLayout(self):
        """
        This function determines the location of reference, control and data resource elements (Tones) for the desired OFDM symbol
        """
        # Layout the tone indices for each subcarrier
        self.LayoutFirstReferenceSymbol[0,:]  = np.arange(-self.NumHalfSubcarriers, self.NumHalfSubcarriers + 1, 1, np.uint8)
        self.LayoutDataSymbols[0,:]           = np.arange(-self.NumHalfSubcarriers, self.NumHalfSubcarriers + 1, 1, np.uint8)
        self.LayoutOtherReferenceSymbols[0,:] = np.arange(-self.NumHalfSubcarriers, self.NumHalfSubcarriers + 1, 1, np.uint8)

        # Layout the pilot tones and control information
        self.LayoutFirstReferenceSymbol [1, self.NumSubcarriers-1]  = CFlexLinkConfig.DmrsReferenceResourceElementP0
        self.LayoutOtherReferenceSymbols[1, self.NumSubcarriers-1]  = CFlexLinkConfig.DmrsReferenceResourceElementP0
        for T in range(0, self.NumTriplets):
            Index1 = T*3
            Index2 = T*3 + 1
            Index3 = T*3 + 2
            self.LayoutFirstReferenceSymbol [1, Index1]  = CFlexLinkConfig.DmrsReferenceResourceElementP0
            self.LayoutFirstReferenceSymbol [1, Index3]  = CFlexLinkConfig.ControlResourceElement
            if self.ControlInfo.NumberOfTxAntennaPorts == 2:
                self.LayoutFirstReferenceSymbol [1, Index2]  = CFlexLinkConfig.DmrsReferenceResourceElementP1
            else:
                self.LayoutFirstReferenceSymbol [1, Index2]  = CFlexLinkConfig.ReservedResourceElement

            self.LayoutOtherReferenceSymbols[1, Index1]  = CFlexLinkConfig.DmrsReferenceResourceElementP0
            self.LayoutOtherReferenceSymbols[1, Index3]  = CFlexLinkConfig.DataResourceElement
            if self.ControlInfo.NumberOfTxAntennaPorts == 2:
                self.LayoutOtherReferenceSymbols[1, Index2]  = CFlexLinkConfig.DmrsReferenceResourceElementP1
            else:
                self.LayoutOtherReferenceSymbols[1, Index2]  = CFlexLinkConfig.DataResourceElement
            
        # Layout of Data Symbols
        self.LayoutDataSymbols[1,:]           = CFlexLinkConfig.DataResourceElement * np.ones(self.NumSubcarriers) 
        if self.ControlInfo.Phi == True:
            TonesPhiP0 = [-360, -270, -180, -90, 90, 180, 270, 360]
            k          = np.array(TonesPhiP0, np.int16) + self.NumHalfSubcarriers
            self.LayoutDataSymbols[1, k] = CFlexLinkConfig.PhaseReferenceResourceElementP0 * np.ones(8, np.uint8)
            if self.ControlInfo.NumberOfTxAntennaPorts == 2:
                TonesPhiP1 = [-361, -271, -181, -91, 89, 179, 269, 359]
                k          = np.array(TonesPhiP1, np.int16) + self.NumHalfSubcarriers
                self.LayoutDataSymbols[1, k] = CFlexLinkConfig.PhaseReferenceResourceElementP1 * np.ones(8, np.uint8)

        # Layout of Dc Subcarriers
        self.LayoutFirstReferenceSymbol[1, self.NumHalfSubcarriers]  = CFlexLinkConfig.ReservedResourceElement
        self.LayoutOtherReferenceSymbols[1, self.NumHalfSubcarriers] = CFlexLinkConfig.ReservedResourceElement
        self.LayoutDataSymbols[1, self.NumHalfSubcarriers]           = CFlexLinkConfig.ReservedResourceElement

        # Compute the number of resource elements used for data QAM value in the other reference symbols and the data symbols
        self.DataResInDataSymbols     = 0
        self.DataResInOtherRefSymbols = 0

        for ResourceElementIndex in range(0, len(self.LayoutDataSymbols)):
            if self.LayoutOtherReferenceSymbols[1, ResourceElementIndex] == CFlexLinkConfig.DataResourceElement:
                self.DataResInOtherRefSymbols += 1
            if self.LayoutDataSymbols[1, ResourceElementIndex]           == CFlexLinkConfig.DataResourceElement:
                self.DataResInDataSymbols += 1


                

    # ---------------------------------------------------
    # > Member function: Plot ResourceGrid Layout
    # ---------------------------------------------------
    def GetResourceGridLayout(self
                             , bPlot: bool        = False
                             , NumberSymbols: int = 40):
        '''
        This function will generate and potentially plot the resource grid containing the type of RE
        '''
        # Error checking
        assert isinstance(bPlot, bool)
        assert isinstance(NumberSymbols, int)
        assert NumberSymbols > 0 and NumberSymbols < 10000

        # Create and program the ResourceGrid
        self.ResourceGrid       = np.zeros([self.NumSubcarriers, NumberSymbols], np.uint8)
        self.ResourceGrid[:, 0] = self.LayoutFirstReferenceSymbol[1, :]

        for SymbolIndex in range(1, NumberSymbols):
            if (SymbolIndex % self.ControlInfo.ReferenceSymbolPeriodicity) == 0:
                self.ResourceGrid[:, SymbolIndex] = self.LayoutOtherReferenceSymbols[1, :]
            else:
                self.ResourceGrid[:, SymbolIndex] = self.LayoutDataSymbols[1, :]

        # Plot the resource grid if needed
        if bPlot == True:
            # In order to get the right colors to plot, I had to change the values in the self.ResourceGrid
            # to go with the Set1 color map
            rows, columns = self.ResourceGrid.shape
            ColorGrid     = np.zeros([rows, columns], np.uint8)
            for row in range(0, rows):
                for column in range(0, columns):
                    Value = self.ResourceGrid[row, column]
                    if Value == 0: ColorGrid[row, column] = 1
                    if Value == 1: ColorGrid[row, column] = 4
                    if Value == 2: ColorGrid[row, column] = 5
                    if Value == 3: ColorGrid[row, column] = 8
                    if Value == 4: ColorGrid[row, column] = 6
                    if Value == 5: ColorGrid[row, column] = 3
                    if Value == 6: ColorGrid[row, column] = 11

            cmap = plt.cm.Set3 
            plt.figure(1)            
            plt.pcolor(ColorGrid,  cmap='Set3' )
            plt.title('FlexLink Resource Grid')
            plt.ylabel('Subcarrier index k')
            plt.xlabel('OFDM symbol Index')
            plt.grid(color='#999999') 
            plt.colorbar()
            plt.show()
           # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib






# -------------------------------------------------------------
# Test bench
# -------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------
    # Exercising the Enum classes
    # --------------------------------
    EBw = EBandwidth.Bw20MHz
    print(EBw.name + ' = ' + str(EBw.value))
    ESc = ESubcarrierSpacing.Sc20KHz
    print(ESc.name + ' = ' + str(ESc.value))
    ECp = ECyclicPrefix.Ec4MicroSec
    print(ECp.name + ' = ' + str(ECp.value))
    ERe = EReType.DmrsPort0 
    print(ERe.name + ' = ' + str(ERe.value))

    # ---------------------------------
    # Testing the CResourceElement class
    # ---------------------------------
    A = CResourceElement.CreateReArray(FreqUnitArray        = [0, 1, 2, 3, 4, 5]
                                     , TimeUnitArray        = [0, 1]
                                     , Type                 = EReType.DmrsPort0)



    # --------------------------------
    # Excercise the CControlInformation class
    # --------------------------------
    ControlInfo = CControlInformation(ESc = ESubcarrierSpacing.Sc20KHz   # -> ESc = The subcarrier spacing
                                    , Pi  = 3   # Pi  = 0/1.. /15  -> Pi  = Index for the spacing in OFDM symbols between reference sybmols. 
                                    , Ni  = 2   # Ni  = 0/1/2/3    -> Ni  = Index for number of Ofdm symbols in the signal field
                                    , Ai  = 1   # Ai  = 0/1        -> Ai  = Index for number of Tx Antenna ports for which reference signals are available.                      
                                    , Phi = 1   # Phi = 0/1        -> Phi = No / Yes Phase reference signals  
                                    , Bi  = 1   # Ri  = 0/1        -> Bi  = No / Yes Reference signals boosting by sqrt(2) in magnitude
                                    , Qi  = 1)  # Qi  = 0/1  

    print(str(ControlInfo))
    print(' ')

    # --------------------------------
    # Excercise the CSignalField class
    # --------------------------------  
    SignalField = CSignalField(EBS1i = 0     # AvailableTransportBlockSizes = [648, 1296, 1944, 0] for Payload 1
                             , NTBS1 = 100   # Number of transport block sizes 0 ... 65535
                             , FEC1i = 0     # AvailableFecOptions          = ['LDPC_1_2', 'LDPC_2_3', 'LDPC_3_4', 'LDPC_5_6'] for Payload 1
                             , RM1i  = 2     # AvailableRateMatchingOptions = ['0', '25', '50', '100']       for Payload 1
                             , BPS1i = 1     # AvailableBitsPerSymbol       = [1, 2, 4, 6]     for Payload 1
                             , EBS2i = 0     # AvailableTransportBlockSizes = [648, 1296, 1944, 0] for Payload 2
                             , NTBS2 = 100   # Number of transport block sizes 0 ... 65525
                             , FEC2i = 0     # AvailableFecOptions          = ['LDPC_1_2', 'LDPC_2_3', 'LDPC_3_4', 'LDPC_5_6'] for Payload 2
                             , RM2i  = 0     # AvailableRateMatchingOptions = ['0', '25', '50', '100']        for Payload 2
                             , BPS2i = 3     # AvailableBitsPerSymbol       = [1, 2, 4, 6]         for Payload 2
                             , UserBits = [0] * 24)

    print(str(SignalField))


    # --------------------------------
    # Excercise the CFlexConfiguration class
    # -------------------------------- 
    BaseSampleRate    = 20e6
    OverSamplingRatio = 2
    FlexLinkConfig = CFlexLinkConfig( EBandwidth.Bw20MHz
                                    , ESubcarrierSpacing.Sc20KHz
                                    , ECyclicPrefix.Ec4MicroSec
                                    , ControlInfo
                                    , SignalField
                                    , BaseSampleRate
                                    , OverSamplingRatio)

    FlexLinkConfig.GetResourceGridLayout(bPlot = True
                                       , NumberSymbols = 40)

