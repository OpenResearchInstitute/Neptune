# File:       FlexLinkTransmitter.py
# Notes:      This script unifies the transmit chain of the FlexLink modem

__title__     = "FlexLinkTransmitter"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Dec, 26th, 2022"
__copyright__ = 'Andreas Schwarzinger'
__license__   = "MIT"

# --------------------------------------------------------
# Import Statements
# --------------------------------------------------------
# was from   FlexLinkParameters import *
import sys
import os

# this assumes that you are running from a path located where the
# code is also located. 
# Specific directory name you want to change to
working_directory = "python"

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


# chat [9:12 AM] Schwarzinger Andreas (8USNT)
# --------------------------------------------
# Import Modules
# --------------------------------------------
"""
import os
import sys                               # We use sys to include new search paths of modules
OriginalWorkingDirectory = os.getcwd()   # Get the current directory
DirectoryOfThisFile      = os.path.dirname(__file__)   # FileName = os.path.basename
if DirectoryOfThisFile != '':
    os.chdir(DirectoryOfThisFile)        # Restore the current directory
 
# There are modules that we will want to access and they sit two directories above DirectoryOfThisFile
sys.path.append(DirectoryOfThisFile + "\\..\\..\\DspComm")
"""



import FlexLinkParameters as fp
from   FlexLinkCoder      import CCrcProcessor, CLdpcProcessor
from   QamMapping         import CQamMappingIEEE
import Preamble
import numpy              as np
import math
import matplotlib.pyplot  as plt

# to add this file path to vscode use "ctrl + ," then search to extrapaths then add the path
# pylance or interpreter can now find the libs
import Visualization   


# --------------------------------------------------------
# > Class: CFlexTransmitter
# --------------------------------------------------------
class CFlexTransmitter():
    '''
    brief: A class that unifies the FlexLink transmitter portion of the modem
    '''
    # -----------------------------------
    # General class attributes
    # -----------------------------------
    NumCrcBits = 16

    # ----------------------------------
    # > Function: Constructor
    # ----------------------------------
    def __init__(self
               , Configuration: fp.CFlexLinkConfig):
        
        # Error checking
        assert isinstance(Configuration, fp.CFlexLinkConfig)  
   
        # Reference some of the parameters easy access
        self.Configuration  = Configuration
        self.SampleRate     = Configuration.BaseSampleRate #was: SampleRate
        self.FftSize        = Configuration.FftSize
        self.ScSpacing      = Configuration.SubcarrierSpacing
        self.NumSubcarriers = Configuration.NumSubcarriers
        self.CpSamples      = Configuration.CpDurationSamples
        self.OccupiedBw     = self.NumSubcarriers * self.ScSpacing


        # Build the preamble, these are constant depending on sample rate
        # Build the AGC Burst
        self.AgcBurst = Preamble.GenerateAgcBurst(self.SampleRate)
        # Build the PreambleA
        self.PreambleA = Preamble.GeneratePreambleA(self.SampleRate, 1024, 'long')
        # Build the PreambleB
        self.PreambleB = Preamble.GeneratePreambleB()

        # Ensure that the actual subcarrier spacing is close to the one provided by the
        # CControlInfo instance. If the sample rate is not 20.48MHz but for example
        # 20MHz or 19.2MHz, then the actual subcarrier spacing won't be exactly 20KHz/40KHz/80KHz/160KHz
        assert abs(self.ScSpacing - Configuration.ControlInfo.ESc.value) < 1000


    # -------------------------------------------------------------------------
    # > Function: BuildTxWaveform for Point to Point transmit signal
    # -------------------------------------------------------------------------
    def BuildTxWaveform(self
                      , bFreqOffsetSync: bool = False
                      , PayloadA: bytearray = bytearray(1)
                      , PayloadB: bytearray = bytearray(1)):
        '''
        In this function we construct the waveform piece by piece
        '''
        # ----------------------------------------------
        # Type and Error Checking
        # ----------------------------------------------
        assert isinstance(bFreqOffsetSync, bool)
        assert isinstance(PayloadA, bytearray)
        assert isinstance(PayloadB, bytearray)

        # Error checking (we mostly want to ensure that the size of the payload are correct)
        NumBytesProvidedPayloadA = len(PayloadA)
        NumOfMacDataBitsPayloadA = self.Configuration.SignalField.NumberOfMacDataBits1
        assert math.ceil(NumOfMacDataBitsPayloadA / 8) == NumBytesProvidedPayloadA

        NumBytesProvidedPayloadB = len(PayloadB)
        NumOfMacDataBitsPayloadB = self.Configuration.SignalField.NumberOfMacDataBits2
        assert math.ceil(NumOfMacDataBitsPayloadB / 8) == NumBytesProvidedPayloadB

        # -------------------------------------------------
        # Find the number of OFDM symbols that we need
        # -------------------------------------------------
        # A. Determine the number of OFDM symbol that we need to map both PayloadA and PayloadB
        self.NumOfdmSymbols  = 0
        # The number of fixed OFDM symbols are the first reference symbol and the signal field symbols
        self.NumOfdmSymbols += 1
        self.NumOfdmSymbols += self.Configuration.ControlInfo.NumberOfSignalFieldSymbols

        AccountedForAllQamSymbols = False
        DataQamSymbolsUnaccounted = self.Configuration.SignalField.NumberQamSymbols1 + \
                                    self.Configuration.SignalField.NumberQamSymbols2

        # Increment the number of OFDM symbols until all Data Qam Symbols are accounted for
        while AccountedForAllQamSymbols == False:
            CurrentSymbol = self.NumOfdmSymbols
            if CurrentSymbol % self.Configuration.ControlInfo.ReferenceSymbolPeriodicity == 0:
                # This is a reference symbol
                AvailableDataReInOfdmSymbol = self.Configuration.DataResInOtherRefSymbols
            else:
                # This is a pure data symbol
                AvailableDataReInOfdmSymbol = self.Configuration.DataResInDataSymbols

            DataQamSymbolsUnaccounted -= AvailableDataReInOfdmSymbol

            if DataQamSymbolsUnaccounted > 0:
                self.NumOfdmSymbols += 1
            else:
                AccountedForAllQamSymbols = True

        self.ResourceGrid = np.zeros([self.NumSubcarriers, self.NumOfdmSymbols], dtype = np.complex128)



        # ----------------------------------------------
        # # Build the preamble
        # # ----------------------------------------------
        # # Build the AGC Burst
        # AgcBurst = Preamble.GenerateAgcBurst(self.SampleRate)
        # SampleLengthAgcBurst = len(AgcBurst)

        # # Build the PreambleA
        # PreambleA = Preamble.GeneratePreambleA(self.SampleRate
        #                                     , 'long')
        # SampleLengthPreambleA = len(PreambleA)

        # # Build the PreambleB
        # PreambleB = Preamble.GeneratePreambleB()
        # SampleLengthPreambleB = len(PreambleB)
        
        
        print('Sample Length AgcBurst = ' + str(len(self.AgcBurst)))
        print('Sample Length PreambleA = ' + str(len(self.PreambleA)))
        print('Sample Length PreambleB = ' + str(len(self.PreambleB)))
        
        # Build the Payload
        SampleLengthPayloads  = self.NumOfdmSymobls * (self.FftSize + self.CpSamples)





    # --------------------------------------------------
    # > Member Method: Encode Payload
    # --------------------------------------------------
    def EncodePayload(self
                    , PayloadType:     str
                    , MacBytePayload:  bytearray):   
        '''
        brief: This function encodes the MacPayload Bits for the given payload
        param: PayloadType  (Input)   -> Either 'A', or 'B'
        param: MacBitsPaylaod (Input) -> A bytearray
        '''
        # --------------------------------------------
        # Type and Error Checking
        # --------------------------------------------
        assert isinstance(PayloadType, str)
        assert PayloadType.upper() == 'A' or PayloadType.upper() == 'B'
        assert isinstance(MacBytePayload, bytearray)

        # Transfer parameters for the payload to be encoded
        if PayloadType.upper() == 'A':
            # MBS = MAC Block Size is the transport block size - 16 crc bits           
            MBS                = self.Configuration.SignalField.TBS1 - fp.CSignalField.NumberCrcBits
            EBS                = self.Configuration.SignalField.EBS1 
            NumTransportBlocks = self.Configuration.SignalField.NTB1
            FecMethod          = self.Configuration.SignalField.FEC1
            RateMatchMult      = self.Configuration.SignalField.RM1
            BitsPerSymbols     = self.Configuration.SignalField.BPS1
        else:
            # MBS = MAC Block Size is the transport block size - 16 crc bits     
            MBS                = self.Configuration.SignalField.TBS2 - fp.CSignalField.NumberCrcBits
            EBS                = self.Configuration.SignalField.EBS2
            NumTransportBlocks = self.Configuration.SignalField.NTB2
            FecMethod          = self.Configuration.SignalField.FEC2
            RateMatchMult      = self.Configuration.SignalField.RM2
            BitsPerSymbols     = self.Configuration.SignalField.BPS2             

        # Ensure that the payload provided in the function argument list is correctly sized
        BitsInPayload = len(MacBytePayload) * 8
        NumTransportBlocksInGivenPayload = math.floor(BitsInPayload/MBS)
        assert NumTransportBlocks == NumTransportBlocksInGivenPayload, 'Invalid payload lenth'
        assert BitsInPayload - NumTransportBlocks * MBS < 8 and BitsInPayload - NumTransportBlocks * MBS >= 0

        # ----------------------------------------------
        # Extracting the MAC blocks from the bytearray payload
        # ----------------------------------------------
        ListOfMacBlocks = []
        SingleMacBlock  = np.zeros(MBS, np.uint8)
        BitIndex        = 0
        for Byte in MacBytePayload:
            # We are done as we have assembled all needed MacBlocks
            if len(ListOfMacBlocks) == NumTransportBlocks:
                break
            for Index in range(0, 8):
                SingleMacBlock[BitIndex] = (2**(7-Index) & Byte) >> (7-Index)
                BitIndex += 1
                if BitIndex == MBS:
                    # Switch to next Mac Block
                    ListOfMacBlocks.append(SingleMacBlock.copy())
                    SingleMacBlock = np.zeros(MBS, np.uint8)
                    BitIndex       = 0

        # ------------------------------------------------
        # Iterate through each Mac Block and encode it completely
        # ------------------------------------------------
        if FecMethod == 'LDPC_1_2':
            LdpcProcessor  = CLdpcProcessor('WLAN', [], EBS, '1/2')
        elif FecMethod == 'LDPC_2_3':
            LdpcProcessor  = CLdpcProcessor('WLAN', [], EBS, '2/3')
        elif FecMethod == 'LDPC_3_4':   # 
            LdpcProcessor  = CLdpcProcessor('WLAN', [], EBS, '3/4')
        else:
            LdpcProcessor  = CLdpcProcessor('WLAN', [], EBS, '3/4')


        ListOfTransportBlocks   = []
        ListOfEncodedBlocks     = []
        ListOfRateMatchedBlocks = []
        ListOfQamSymbols        = []
        for MacBlock in ListOfMacBlocks:
            # CRC Attachment
            CrcBits = CCrcProcessor.ComputeCrc(16, MacBlock)
            TransportBlock = np.hstack([CrcBits, np.array(MacBlock, dtype = np.uint8)])
            ListOfTransportBlocks.append(TransportBlock.copy())

            # Encode given the right FEC choice
            EncodedBits = LdpcProcessor.EncodeBits(TransportBlock, 'hard')
            ListOfEncodedBlocks.append(EncodedBits.copy())

            # Rate Matching
            RateMatchedBits = np.tile(EncodedBits, RateMatchMult)
            ListOfRateMatchedBlocks.append(RateMatchedBits.copy())


        # QAM Mapping
        QamSymbols = CQamMappingIEEE.Mapping(BitsPerSymbols, RateMatchedBits)
        ListOfQamSymbols.append(QamSymbols.copy())



# ----------------------------------------------------------------------
# > Testbench
# ----------------------------------------------------------------------
if __name__ == '__main__':

    
    # ---------------------------------------------
    # 1. Set the fundamental simulation parameters
    # ---------------------------------------------
    # The following four parameters must be known for any of the setup structures
    # to be properly programmed. For now, it is best not to changes these parameters
    # as a different combination is not yet supported.
    eBandwidth         = fp.EBandwidth.Bw20MHz
    eSubcarrierSpacing = fp.ESubcarrierSpacing.Sc20KHz
    eCyclicPrefix      = fp.ECyclicPrefix.Ec4MicroSec
    BaseSampleRate     = 20.48e6

    # ---------------------------------------------
    # 2. Set the CControlInformation Instance
    # ---------------------------------------------
    # In order to see what these parameters actual translate to, just uncomment the >print(ControlInfo) lines below
    ControlInfo = fp.CControlInformation(ESc = eSubcarrierSpacing   # -> ESc = The subcarrier spacing
                                    , Pi  = 3   # Pi  = 0/1.. /15  -> Pi  = Index for the spacing in OFDM symbols between reference sybmols. 
                                    , Ni  = 2   # Ni  = 0/1/2/3    -> Ni  = Index for number of Ofdm symbols in the signal field
                                    , Ai  = 1   # Ai  = 0/1        -> Ai  = Index for number of Tx Antenna ports for which reference signals are available.                      
                                    , Phi = 0   # Phi = 0/1        -> Phi = No / Yes Phase reference signals  
                                    , Bi  = 1   # Ri  = 0/1        -> Bi  = No / Yes Reference signals boosting by sqrt(2) in magnitude
                                    , Qi  = 1)  # Qi  = 0/1        -> Qi  = 0  /  1  -> BPSK / QPSK SignalField
    
    print(ControlInfo)

    # ---------------------------------------------
    # 3. Set the CSignalField Instance
    # --------------------------------------------- 
    # In order to see what these parameters actual translate to, just uncomment the >print(ControlInfo) lines below
    SignalField = fp.CSignalField(EBS1i = 0     # AvailableTransportBlockSizes = [648, 1296, 1944, 0] for Payload 1
                             , NTBS1 = 100   # Number of transport blocks to be transmitted 0 ... 65535
                             , FEC1i = 0     # AvailableFecOptions          = ['LDPC_1_2', 'LDPC_2_3', 'LDPC_3_4', 'n/a''] for Payload 1
                             , RM1i  = 2     # AvailableRateMatchingOptions = [1, 1.5, 2, 3, 4, 6, 8, 16]      bit multiplier for Payload 1
                             , BPS1i = 1     # AvailableBitsPerSymbol       = [1, 2, 4, 6]     for Payload 1
                             , EBS2i = 0     # AvailableTransportBlockSizes = [648, 1296, 1944, 0] for Payload 2
                             , NTBS2 = 100   # Number of transport block to be transmitted 0 ... 65525
                             , FEC2i = 0     # AvailableFecOptions          = ['LDPC_1_2', 'LDPC_2_3', 'LDPC_3_4', 'n/a'] for Payload 2
                             , RM2i  = 0     # AvailableRateMatchingOptions = [1, 1.5, 2, 3, 4, 6, 8, 16]      bit multiplier for Payload 2
                             , BPS2i = 3     # AvailableBitsPerSymbol       = [1, 2, 4, 6]         for Payload 2
                             , UserBits = [0] * 24)

    print(SignalField)

    # -----------------------------------------------
    # 4. Put it all together as a CFlexLinkConfig Instance
    # -----------------------------------------------
    # This instance should be handed to both the transmitter and receiver.
    # Outside of the data to transmit, the transmitter and receiver should need nothing else.
    OversamplingRatio = 2
    FlexLinkConfig = fp.CFlexLinkConfig( eBandwidth
                                    , eSubcarrierSpacing
                                    , eCyclicPrefix
                                    , ControlInfo
                                    , SignalField
                                    , BaseSampleRate
                                    , OversamplingRatio)  # Actual sample rate = BaseSampleRate * OversamplingRatio

    # Construct the CFlexTransmitter
    Transmitter = CFlexTransmitter(FlexLinkConfig)

    # --------------------------------------------
    # 5. Build up the two payloads
    # --------------------------------------------
    # First, we need to figure out how many mac bits each payload must deliver

    NumMacBitsA      = SignalField.NumberOfMacDataBits1
    NumBytesPayloadA = math.ceil(NumMacBitsA/8)

    NumMacBitsB      = SignalField.NumberOfMacDataBits2
    NumBytesPayloadB = math.ceil(NumMacBitsB/8)

    r = np.random.RandomState(1234) 
    MacBitsPayloadA  = r.randint(0, 256, NumBytesPayloadA, dtype = np.uint8)
    MacBitsPayloadB  = r.randint(0, 256, NumBytesPayloadB, dtype = np.uint8)

    # We now need to convert the bit streams into a byte array.
    ByteArrayPayloadA = bytearray(list(MacBitsPayloadA))
    ByteArrayPayloadB = bytearray(list(MacBitsPayloadB))

    # ----------------------------------------------
    # 6. Encode the MacBits Byte arrays
    # ----------------------------------------------
    EncodedPayloadA = Transmitter.EncodePayload(PayloadType    = 'A'
                                              , MacBytePayload =  ByteArrayPayloadA)   

    EncodedPayloadB = Transmitter.EncodePayload(PayloadType    = 'B'
                                              , MacBytePayload =  ByteArrayPayloadA)   

    
    v = Visualization.Visualization()
    AgcBurst = list(Transmitter.AgcBurst)
    v.plot_constellation(AgcBurst, start=-110, end=-10, name='AGC burst')
    v.plot_iq_data2x(Transmitter.AgcBurst.real, Transmitter.AgcBurst.imag, "agc real", "agc imag")


    Stop =1 


 