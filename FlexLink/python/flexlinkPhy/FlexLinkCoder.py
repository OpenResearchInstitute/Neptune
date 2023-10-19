# File:     FlexLinkCoder.py
# Notes:    This file provides CRC, and FEC encoding and decoding services
# Refences: Digital Signal Processing in Modern Communication Systems (Edition 2) Sections 5.6.1 and 5.6.6
#           IEEE Std 802.11n-2009 (Oct 2009) Amendment 5: Enhancements For Higher Throughput, New York, NY 
#           IEEE Std 802.11-2012

__title__     = "FlexLinkCoder"
__author__    = "Andreas Schwarzinger"
__status__    = "released Version 1.0"
__date__      = "Jan, 4rd, 2023"
__copyright__ = 'Andreas Schwarzinger'

import numpy as np
# was: from   SignalProcessing import *
import SignalProcessing as sp
import DebugUtility
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# > CCrcProcessor Class
# ------------------------------------------------------------------
class CCrcProcessor():
    """
    This class provides cyclic redundancy check calculations for the FlexLink Specification.
    """

    # FlexLink used 2 types of CRC lengths.
    # Length 16 is used to protect the signal field (LTE gcrc16)
    # Length 24 is used to protect each transport block in the payload (LTE gcrc24a)
    Generator24 = np.array([1,1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1], dtype = np.uint8)
    Generator16 = np.array([1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1], dtype = np.uint8)

    # -------------------------------------------------
    @classmethod
    def ComputeCrc(cls
                 , GeneratorSize
                 , InputBitVector) -> np.ndarray:

        # -------------------------------------------------------
        # Error checking
        assert isinstance(GeneratorSize, int),             'The GeneratorSize must be of type int.'
        assert GeneratorSize == 16 or GeneratorSize == 24, 'The GeneratorSize must be either 16 or 24'
        if isinstance(InputBitVector, list):
            bProperBits = all([( (x == 1 or x == 0) and isinstance(x, int)) for x in InputBitVector])
            assert bProperBits, 'The InputBitVector must be composed of 1s and 0s'
            InputBitVector = np.array(InputBitVector, dtype = np.uint8)
        else:
            assert isinstance(InputBitVector, np.ndarray), 'The InputBitVector must be either a list or of type np.ndarray'
            bProperBits = all([(x == 1 or x == 0) for x in InputBitVector])
            assert bProperBits, 'The InputBitVector must be composed of 1s and 0s'              
            InputBitVector = InputBitVector.astype(np.uint8)

        # -----------------------------------------------------
        # Compute the cyclic redundancy check
        if GeneratorSize == 16:
            Generator = cls.Generator16
        else:
            Generator = cls.Generator24

        TempMessage = np.hstack([InputBitVector, np.zeros(GeneratorSize, dtype = np.uint8)])
        for Index in range(0, len(InputBitVector)):
            Range = np.arange(Index, Index + GeneratorSize + 1, 1, dtype = np.int32)
            if TempMessage[Index] == 1:
                TempMessage[Range] = np.mod(TempMessage[Range] + Generator, 2)
            if np.sum(TempMessage) == 0:
                break

        return TempMessage[-GeneratorSize:]           












# ------------------------------------------------------------------
# > CLdpcProcessor Class
# ------------------------------------------------------------------
class CLdpcProcessor():
    """
    brief: This class provides LDPC encoding and decoding services for the FlexLink Specification 
    notes: See IEEE Std 802.11-2012 Annex F
    """
    # -------------------------------
    # Matrix Prototypes for codeword block length n = 648 bits, with subblock size Z = 27 bits
    # -------------------------------
    PrototypeM648_1_2 = np.array([[ 0, -1, -1, -1,  0,  0, -1, -1,  0, -1, -1,  0,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                  [22,  0, -1, -1, 17, -1,  0,  0, 12, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                  [ 6, -1,  0, -1, 10, -1, -1, -1, 24, -1,  0, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1], 
                                  [ 2, -1, -1,  0, 20, -1, -1, -1, 25,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1], 
                                  [23, -1, -1, -1,  3, -1, -1, -1,  0, -1,  9, 11, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1], 
                                  [24, -1, 23,  1, 17, -1,  3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                  [25, -1, -1, -1,  8, -1, -1, -1,  7, 18, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                  [13, 24, -1, -1,  0, -1,  8, -1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                  [ 7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1], 
                                  [11, -1, -1, -1, 19, -1, -1, -1, 13, -1,  3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                  [25, -1,  8, -1, 23, 18, -1, 14,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0], 
                                  [ 3, -1, -1, -1, 16, -1, -1,  2, 25,  5, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM648_2_3 = np.array([[25, 26, 14, -1, 20, -1,  2, -1,  4, -1, -1,  8, -1, 16, -1, 18,  1,  0, -1, -1, -1, -1, -1, -1], 
                                  [10,  9, 15, 11, -1,  0, -1,  1, -1, -1, 18, -1,  8, -1, 10, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                  [16,  2, 20, 26, 21, -1,  6, -1,  1, 26, -1,  7, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                  [10, 13,  5,  0, -1,  3, -1,  7, -1, -1, 26, -1, -1, 13, -1, 16, -1, -1, -1,  0,  0, -1, -1, -1], 
                                  [23, 14, 24, -1, 12, -1, 19, -1, 17, -1, -1, -1, 20, -1, 21, -1,  0, -1, -1, -1,  0,  0, -1, -1], 
                                  [ 6, 22,  9, 20, -1, 25, -1, 17, -1,  8, -1, 14, -1, 18, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                  [14, 23, 21, 11, 20, -1, 24, -1, 18, -1, 19, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1,  0,  0], 
                                  [17, 11, 11, 20, -1, 21, -1, 26, -1,  3, -1, -1, 18, -1, 26, -1,  1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM648_3_4 = np.array([[16, 17, 22, 24,  9,  3, 14, -1,  4,  2,  7, -1, 26, -1,  2, -1, 21, -1,  1,  0, -1, -1, -1, -1], 
                                  [25, 12, 12,  3,  3, 26,  6, 21, -1, 15, 22, -1, 15, -1,  4, -1, -1, 16, -1,  0,  0, -1, -1, -1], 
                                  [25, 18, 26, 16, 22, 23,  9, -1,  0, -1,  4, -1,  4, -1,  8, 23, 11, -1, -1, -1,  0,  0, -1, -1], 
                                  [ 9,  7,  0,  1, 17, -1, -1,  7,  3, -1,  3, 23, -1, 16, -1, -1, 21, -1,  0, -1, -1,  0,  0, -1], 
                                  [24,  5, 26,  7,  1, -1, -1, 15, 24, 15, -1,  8, -1, 13, -1, 13, -1, 11, -1, -1, -1, -1,  0,  0], 
                                  [ 2,  2, 19, 14, 24,  1, 15, 19, -1, 21, -1,  2, -1, 24, -1,  3, -1,  2,  1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM648_5_6 = np.array([[17, 13,  8, 21,  9,  3, 18, 12, 10,  0,  4, 15, 19,  2,  5, 10, 26, 19, 13, 13,  1,  0, -1, -1],
                                  [ 3, 12, 11, 14, 11, 25,  5, 18,  0,  9,  2, 26, 26, 10, 24,  7, 14, 20,  4,  2, -1,  0,  0, -1],
                                  [22, 16,  4,  3, 10, 21, 12,  5, 21, 14, 19,  5, -1,  8,  5, 18, 11,  5,  5, 15,  0, -1,  0,  0],
                                  [ 7,  7, 14, 14,  4, 16, 16, 24, 24, 10,  1,  7, 15,  6, 10, 26,  8, 18, 21, 14,  1, -1, -1,  0]], dtype = np.int8)


    # -------------------------------
    # Matrix Prototypes for codeword block length n = 1296 bits, with subblock size Z = 54 bits
    # -------------------------------
    PrototypeM1296_1_2 = np.array([[40, -1, -1, -1, 22, -1, 49, 23, 43, -1, -1, -1,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [50,  1, -1, -1, 48, 35, -1, -1, 13, -1, 30, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [39, 50, -1, -1,  4, -1,  2, -1, -1, -1, -1, 49, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [33, -1, -1, 38, 37, -1, -1,  4,  1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1], 
                                   [45, -1, -1, -1,  0, 22, -1, -1, 20, 42, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1], 
                                   [51, -1, -1, 48, 35, -1, -1, -1, 44, -1, 18, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                   [47, 11, -1, -1, -1, 17, -1, -1, 51, -1, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                   [ 5, -1, 25, -1,  6, -1, 45, -1, 13, 40, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [33, -1, -1, 34, 24, -1, -1, -1, 23, -1, -1, 46, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1], 
                                   [ 1, -1, 27, -1,  1, -1, -1, -1, 38, -1, 44, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                   [-1, 18, -1, -1, 23, -1, -1,  8,  0, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0], 
                                   [49, -1, 17, -1, 30, -1, -1, -1, 34, -1, -1, 19,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1296_2_3 = np.array([[25, 52, 41,  2,  6, -1, 14, -1, 34, -1, -1, -1, 24, -1, 37, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                   [43, 31, 29,  0, 21, -1, 28, -1, -1,  2, -1, -1,  7, -1, 17, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                   [20, 33, 48, -1,  4, 13, -1, 26, -1, -1, 22, -1, -1, 46, 42, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [45,  7, 18, 51, 12, 25, -1, -1, -1, 50, -1, -1,  5, -1, -1, -1,  0, -1, -1, -1,  0,  0, -1, -1], 
                                   [35, 40, 32, 16,  5, -1, -1, 18, -1, -1, 43, 51, -1, 32, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                   [ 9, 24, 13, 22, 28, -1, -1, 37, -1, -1, 25, -1, -1, 52, -1, 13, -1, -1, -1, -1, -1, -1,  0,  0], 
                                   [32, 22,  4, 21, 16, -1, -1, -1, 27, 28, -1, 38, -1, -1, -1,  8,  1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1296_3_4 = np.array([[39, 40, 51, 41,  3, 29,  8, 36, -1, 14, -1,  6, -1, 33, -1, 11, -1,  4,  1,  0, -1, -1, -1, -1], 
                                   [48, 21, 47,  9, 48, 35, 51, -1, 38, -1, 28, -1, 34, -1, 50, -1, 50, -1, -1,  0,  0, -1, -1, -1], 
                                   [30, 39, 28, 42, 50, 39,  5, 17, -1,  6, -1, 18, -1, 20, -1, 15, -1, 40, -1, -1,  0,  0, -1, -1], 
                                   [29,  0,  1, 43, 36, 30, 47, -1, 49, -1, 47, -1,  3, -1, 35, -1, 34, -1,  0, -1, -1,  0,  0, -1], 
                                   [ 1, 32, 11, 23, 10, 44, 12,  7, -1, 48, -1,  4, -1,  9, -1, 17, -1, 16, -1, -1, -1, -1,  0,  0], 
                                   [13,  7, 15, 47, 23, 16, 47, -1, 43, -1, 29, -1, 52, -1,  2, -1, 53, -1,  1, -1, -1, -1, -1,  0]], dtype = np.int8)
     
    PrototypeM1296_5_6 = np.array([[48, 29, 37, 52,  2, 16,  6, 14, 53, 31, 34,  5, 18, 42, 53, 31, 45, -1, 46, 52,  1,  0, -1, -1],
                                   [17,  4, 30,  7, 43, 11, 24,  6, 14, 21,  6, 39, 17, 40, 47,  7, 15, 41, 19, -1, -1,  0,  0, -1],
                                   [ 7,  2, 51, 31, 46, 23, 16, 11, 53, 40, 10,  7, 46, 53, 33, 35, -1, 25, 35, 38,  0, -1,  0,  0],
                                   [19, 48, 41,  1, 10,  7, 36, 47,  5, 29, 52, 52, 31, 10, 26,  6,  3,  2, -1, 51,  1, -1, -1,  0]], dtype = np.int8)

    # -------------------------------
    # Matrix Prototypes for codeword block length n = 1944 bits, with subblock size Z = 81 bits
    # -------------------------------
    PrototypeM1944_1_2 = np.array([[57, -1, -1, -1, 50, -1, 11, -1, 50, -1, 79, -1,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [ 3, -1, 28, -1,  0, -1, -1, -1, 55,  7, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [30, -1, -1, -1, 24, 37, -1, -1, 56, 14, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1], 
                                   [62, 53, -1, -1, 53, -1, -1,  3, 35, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1], 
                                   [40, -1, -1, 20, 66, -1, -1, 22, 28, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1], 
                                   [ 0, -1, -1, -1,  8, -1, 42, -1, 50, -1, -1,  8, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                   [69, 79, 79, -1, -1, -1, 56, -1, 52, -1, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                   [65, -1, -1, -1, 38, 57, -1, -1, 72, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [64, -1, -1, -1, 14, 52, -1, -1, 30, -1, -1, 32, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1], 
                                   [-1, 45, -1, 70,  0, -1, -1, -1, 77,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                   [ 2, 56, -1, 57, 35, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0], 
                                   [24, -1, 61, -1, 60, -1, -1, 27, 51, -1, -1, 16,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1944_2_3 = np.array([[61, 75,  4, 63, 56, -1, -1, -1, -1, -1, -1,  8, -1,  2, 17, 25,  1,  0, -1, -1, -1, -1, -1, -1], 
                                   [56, 74, 77, 20, -1, -1, -1, 64, 24,  4, 67, -1,  7, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1], 
                                   [28, 21, 68, 10,  7, 14, 65, -1, -1, -1, 23, -1, -1, -1, 75, -1, -1, -1,  0,  0, -1, -1, -1, -1], 
                                   [48, 38, 43, 78, 76, -1, -1, -1, -1,  5, 36, -1, 15, 72, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [40,  2, 53, 25, -1, 52, 62, -1, 20, -1, -1, 44, -1, -1, -1, -1,  0, -1, -1, -1,  0,  0, -1, -1], 
                                   [69, 23, 64, 10, 22, -1, 21, -1, -1, -1, -1, -1, 68, 23, 29, -1, -1, -1, -1, -1, -1,  0,  0, -1], 
                                   [12,  0, 68, 20, 55, 61, -1, 40, -1, -1, -1, 52, -1, -1, -1, 44, -1, -1, -1, -1, -1, -1,  0,  0], 
                                   [58,  8, 34, 64, 78, -1, -1, 11, 78, 24, -1, -1, -1, -1, -1, 58,  1, -1, -1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1944_3_4 = np.array([[48, 29, 28, 39,  9, 61, -1, -1, -1, 63, 45, 80, -1, -1, -1, 37, 32, 22,  1,  0, -1, -1, -1, -1], 
                                   [ 4, 49, 42, 48, 11, 30, -1, -1, -1, 49, 17, 41, 37, 15, -1, 54, -1, -1, -1,  0,  0, -1, -1, -1], 
                                   [35, 76, 78, 51, 37, 35, 21, -1, 17, 64, -1, -1, -1, 59,  7, -1, -1, 32, -1, -1,  0,  0, -1, -1], 
                                   [ 9, 65, 44,  9, 54, 56, 73, 34, 42, -1, -1, -1, 35, -1, -1, -1, 46, 39,  0, -1, -1,  0,  0, -1], 
                                   [ 3, 62,  7, 80, 68, 26, -1, 80, 55, -1, 36, -1, 26, -1,  9, -1, 72, -1, -1, -1, -1, -1,  0,  0], 
                                   [26, 75, 33, 21, 69, 59,  3, 38, -1, -1, -1, 35, -1, 62, 36, 26, -1, -1,  1, -1, -1, -1, -1,  0]], dtype = np.int8)

    PrototypeM1944_5_6 = np.array([[13, 48, 80, 66,  4, 74,  7, 30, 76, 52, 37, 60, -1, 49, 73, 31, 74, 73, 23, -1,  1,  0, -1, -1],
                                   [69, 63, 74, 56, 64, 77, 57, 65,  6, 16, 51, -1, 64, -1, 68,  9, 48, 62, 54, 27, -1,  0,  0, -1],
                                   [51, 15,  0, 80, 24, 25, 42, 54, 44, 71, 71,  9, 67, 35, -1, 58, -1, 29, -1, 53,  0, -1,  0,  0],
                                   [16, 29, 36, 41, 44, 56, 59, 37, 50, 24, -1, 65,  4, 65, 52, -1,  4, -1, 73, 52,  1, -1, -1,  0]], dtype = np.int8)
    
    # --------------------------------------------------------------------------------
    # > This function expands the prototype matrix into a parity check matrix
    # --------------------------------------------------------------------------------
    @classmethod
    def CreateParityCheckMatrix(cls
                              , iBlockLength: int = 648
                              , strRate:      str = '1/2') -> np.ndarray:
        """
        brief: This function will create a parity check matrix from one of the available prototype matrices
        param: iBlockLength  - Currently we have a choice of 648, 1296 and 1944
        param: strRate       - Currently we have a choice of '1/2', '2/3', '3/4, '5/6'
        """
        
        # Error checking
        assert isinstance(iBlockLength, int),                                       'The iBlockLength input type is not int.'
        assert isinstance(strRate, str),                                            'The strRate input typ is not str.'
        assert iBlockLength == 648 or iBlockLength == 1296 or iBlockLength == 1944, 'The block length must be 648 or 1296 or 1944.'
        assert strRate == '1/2' or strRate == '2/3' or strRate == '3/4' or strRate == '5/6', \
                                                                           'The strRate input arguement must be 1/2, 2/3, 3/4 or 5/6'

        # Determine current prototype matrix
        match iBlockLength:
            case 648:
                if strRate == '1/2': PrototypeMatrix = cls.PrototypeM648_1_2
                if strRate == '2/3': PrototypeMatrix = cls.PrototypeM648_2_3
                if strRate == '3/4': PrototypeMatrix = cls.PrototypeM648_3_4
                if strRate == '5/6': PrototypeMatrix = cls.PrototypeM648_5_6
                SubmatrixSize = 27
            case 1296:
                if strRate == '1/2': PrototypeMatrix = cls.PrototypeM1296_1_2
                if strRate == '2/3': PrototypeMatrix = cls.PrototypeM1296_2_3
                if strRate == '3/4': PrototypeMatrix = cls.PrototypeM1296_3_4
                if strRate == '5/6': PrototypeMatrix = cls.PrototypeM1296_5_6
                SubmatrixSize = 54
            case 1944:
                if strRate == '1/2': PrototypeMatrix = cls.PrototypeM1944_1_2
                if strRate == '2/3': PrototypeMatrix = cls.PrototypeM1944_2_3
                if strRate == '3/4': PrototypeMatrix = cls.PrototypeM1944_3_4
                if strRate == '5/6': PrototypeMatrix = cls.PrototypeM1944_5_6
                SubmatrixSize = 81
            case _:
                assert False, 'An error has occured.'

        # Build the parity check matrix, H
        Rows, Columns = PrototypeMatrix.shape
        H = np.zeros([Rows * SubmatrixSize, Columns * SubmatrixSize], dtype = np.uint8)

        for row in range(0, Rows):
            for column in range(0, Columns):
                # The cyclic shift
                iCyclicShift = PrototypeMatrix[row, column]
                
                if iCyclicShift == -1:    # Then do not insert the eye matrix
                    continue

                # Create the eye matrix and cyclically shift it
                Eye        = np.eye(SubmatrixSize, dtype = np.uint8)
                EyeShifted = np.roll(Eye, iCyclicShift, axis = 1)   # Cyclic shift of each row

                # Figure out the row/column ranges inside the H matrix where we want to insert the Eye matrix
                RowStart    = row    * SubmatrixSize
                RowStop     = RowStart + SubmatrixSize
                ColumnStart = column * SubmatrixSize 
                ColumnStop  = ColumnStart + SubmatrixSize

                # Insert the Eye matrix
                H[RowStart:RowStop, ColumnStart:ColumnStop] = EyeShifted

        # Return the parity check matrix
        return H




    # --------------------------------------------------------------------------------
    # > This function transform the parity check matrix via the Gauss-Jordan Elimination
    # --------------------------------------------------------------------------------
    @staticmethod
    def TransformParityCheckMatrix(H) -> np.ndarray:
        """
        Transform the parity check matrix into a form that can be used to create the generator matrix
        """
        H_New = H.copy()
        rows, columns = H_New.shape
        N             = columns   # The number of encoded bits
        L             = rows      # The number of parity bits
        K             = N - L     # The number of message bits

        # ------------------------------------
        # Step 1: The forward elimination step
        for column in range(K, N):
            # [RowOfInterest, column] are the coordinates of the diagonal of the LxL square matrix all
            # the way to the right of H_New. The point of the exercise is to place 1's in these positions. 
            # If we don't find a 1 there, we need to look to see whether one of the rows below has a
            # one in this column and then pivot.
            RowOfInterest = column - K    # Increments as follows: 0, 1, 2, 3, 4, ... L - 1

            # 1a. Execute a pivot step if we need it
            if H_New[RowOfInterest, column] != 1:
                # We will now look at the rows below until we find a 1 in this column.
                # Rather than switching the current row with the one containing the 1,
                # we add the row below to the current on. This is fine as we are
                # doing modulo 2 addition. We could have switched them as well.
                RemainingRowsBelow = L - RowOfInterest -1
                bPivotSuccessful  = False
                if RemainingRowsBelow > 0:
                    # We attempt to pivot rows
                    for row in range(RowOfInterest+1, L):
                        if H_New[row, column] == 1:
                            # Modulo two addition of RowOfInterest and row
                            H_New[RowOfInterest, :] = np.remainder(H_New[RowOfInterest, :] + H_New[row,:], 2)
                            bPivotSuccessful = True
                            break
                else:
                    assert bPivotSuccessful, 'The Pivot operation failed. Supply a proper H matrix.' 

            # 1b. Execute the forward elimination step
            #     At the point, we have ensured that the diagonal of the square matrix on the right 
            #     of H_New features ones everywhere
            for row in range(RowOfInterest+1, L):
                if H_New[row, column] == 1:
                    # Modulo two addition of RowOfInterest and row
                    H_New[row, :] = np.remainder(H_New[RowOfInterest, :] + H_New[row,:], 2)

        # ------------------------------------
        # Step 2: The backward elimination step
        for column in range(N-1, K, -1):
            RowOfInterest = column - K   # Decrement as follows: L, L-1, L-2, L-3 ... 2
            # If any of the rows above feature a 1 in this column, get rid of it by adding
            for row in range(0, RowOfInterest):
                if H_New[row, column] == 1:
                    # Modulo two addition of RowOfInterest and row
                    H_New[row, :] = np.remainder(H_New[RowOfInterest, :] + H_New[row,:], 2)

        return H_New



    # ----------------------------------------------------------------------------
    # > This function will compute the generator matrix G
    # ----------------------------------------------------------------------------
    @staticmethod
    def ComputeGeneratorMatrix(H) -> np.ndarray:
        """
        This function computes the generator matrix, G, of a paritycheck matrix, H. 
        """
        rows, columns = H.shape
        N             = columns   # The number of encoded bits
        L             = rows      # The number of parity bits
        K             = N - L     # The number of message bits

        H_Modified = CLdpcProcessor.TransformParityCheckMatrix(H)
        P          = H_Modified[0:L, 0:K]
        PT         = np.transpose(P)
        G          = np.hstack([np.eye(K, dtype = P.dtype), PT])

        # We can verify that the generator matrix was properly computed
        GHT        = np.matmul(G, np.transpose(H))
        GHT_MOD2   = np.remainder(GHT, 2)
        TotalSum   = np.sum(GHT_MOD2[:])
        assert TotalSum == 0, 'The generator was not computed properly.'

        return G



    # ----------------------------------------------------------------------------
    # > This function will compute the SISO Single Parity Check
    # ----------------------------------------------------------------------------
    @staticmethod
    def SISO_SPC_Decoder(rn: np.ndarray) -> np.ndarray:
        '''
        brief: This function implements the single input single output single parity check operation.
        param: rn - This is the vector of intrinsic beliefs at the bit nodes
        var:   l  - This is the vector of extrinsic beliefs for the single paraity check node at index x
        '''
        # Type checking
        assert isinstance(rn, np.ndarray), 'The input rn must be of type nd.ndarray'
        assert np.issubdtype(rn.dtype, np.floating), 'The entries in rn must be floating point numbers'

        # Run the decoder
        l       = np.zeros(len(rn), dtype = rn.dtype) 
        for Index, r in enumerate(rn):
            r_other    = np.delete(rn.copy(), Index)                # Remove the current intrinsic belief from r
            sign_other = -np.sign(np.prod(-r_other))     
            mag_other  = np.min(abs(r_other))
            l[Index]   = sign_other * mag_other            
            
        # Return the new beliefs
        return l





    # ----------------------------------------------------------------------------
    # > The constuctor for a CLdpcProcessor object
    # ----------------------------------------------------------------------------
    def __init__(self
               , strMode:          str = 'WLAN'                         # 'WLAN' or 'CUSTOM'
               , H_Custom:         np.ndarray = np.array([0], np.uint8)       
               , OutputBlockSize:  int = 648
               , strRate:          str = '1/2') -> np.ndarray:

        # Error checking
        assert strMode.lower() == 'wlan' or strMode.lower() == 'custom', 'The strMode argument must be either "wlan" or "custom"'

        if strMode.lower() == 'custom':
            # At this point we only care about the H_Custom argument
            assert isinstance(H_Custom, np.ndarray), 'The H_Custom input argument must be an np.ndarray'
            self.H       = H_Custom
        else:
            # Error checking
            assert isinstance(OutputBlockSize, int),                                             'The iBlockLength input type is not int.'
            assert isinstance(strRate, str),                                                     'The strRate input typ is not str.'
            assert OutputBlockSize == 648 or OutputBlockSize == 1296 or OutputBlockSize == 1944, 'The block length must be 648 or 1296 or 1944.'
            assert strRate == '1/2' or strRate == '2/3' or strRate == '3/4',                     'The strRate input arguement must be 1/2, 2/3, or 3/4.'

            # Record Input arguments
            match strRate:
                case '1/2':
                    self.Rate = 0.5
                case '2/3':  
                    self.Rate = 2/3
                case '3/4':
                    self.Rate = 3/4
                case '5/6':
                    self.Rate = 5/6

            # The number of input bits in a single transport block. Those input bits that can be encoded as one group.
            self.TransportBlockSize = int(OutputBlockSize * self.Rate)

            # The number of output (encoded) bits resulting from the bits in one transport block
            self.OutputBlockSize    = OutputBlockSize
        
            # Create the Parity Check and Generator Matrices 
            self.H                  = CLdpcProcessor.CreateParityCheckMatrix(OutputBlockSize, strRate)
        
        
        # Create the generator matrix
        self.G           = CLdpcProcessor.ComputeGeneratorMatrix(self.H)
        
        rows, columns = self.H.shape
        self.N = self.NumEncodedBits = columns             # The number of encoded bits
        self.L = self.NumParityBits  = rows                # The number of parity bits
        self.K = self.NumMessageBits = self.N - self.L     # The number of message bits

        


    # ----------------------------------------------------------------------------
    # > This function will LDPC encode a vector of input data bits
    # ----------------------------------------------------------------------------
    def EncodeBits(self
                 , InputBits:         np.ndarray
                 , strSoftbitMapping: str  = 'hard'                
                 , bAllowZeroPadding: bool = False) -> np.ndarray:
        """
        brief: This function will create a parity check matrix from one of the available prototype matrices
        param: InputBits         - The input bit vector 
        param: strSoftbitMapping - 'hard' -> 0/1, 'inverting' -> maps 0/1 to 1/-1, 'non-inverting' -> maps 0/1 to -1/1
        param: bAllowZeroPadding - If true, then the input bit vector may be of a size != self.BlockLength 
        """
        
        # ------------------------------
        # Error Checking
        assert isinstance(InputBits, np.ndarray),          'The InputBits input argument must be of type np.ndarray'
        assert np.issubdtype(InputBits.dtype, np.integer), 'The InputBits input argument must be an array of integers'
        assert isinstance(strSoftbitMapping, str)        , 'The strBitMode input argument must be a string'
        assert strSoftbitMapping.lower() == 'hard' or strSoftbitMapping.lower() == 'inverting' \
                                                   or strSoftbitMapping.lower() == 'non-inverting', \
                                                           'The strBitMode input argument is invalid'
        assert isinstance(bAllowZeroPadding, bool),        'The bAllowZeroPadding input arbument must be of type bool'
        assert len(InputBits.shape) == 1 or len(InputBits.shape) == 2, 'The shape of the InputBits input argument is invalid.'

        # The input should be a simple array with shape (NumInputBits, ), not a matrix of dimensions with shape (1, NumInputBits)
        if len(InputBits.shape) == 2:   # Then we must convert the input matrix into a simple array
            InputBits = InputBits[0, :]

        bProperBits = all([((x==0 or x==1)) for x in InputBits])
        assert bProperBits,   'The input bits must be either 0s or 1s.'

        if bAllowZeroPadding == False and len(InputBits) % self.TransportBlockSize != 0:
            assert False, 'The InputBits vector size must be a integer multiple of the transport block size.'
        
        
        # ---------------------------------------------------------------------------
        # Determine number of transport blocks (The number of input bits per block)
        NumberOfInputBits          = len(InputBits)
        if NumberOfInputBits % self.TransportBlockSize > 0:
            NumPaddingBits         = self.TransportBlockSize - NumberOfInputBits % self.TransportBlockSize
        else:
            NumPaddingBits         = 0
            
        ZeroPaddedInputBits        = np.append(InputBits, np.zeros(NumPaddingBits, dtype = InputBits.dtype))
        FinalNumberOfInputBits     = len(ZeroPaddedInputBits)

        assert FinalNumberOfInputBits % self.TransportBlockSize == 0, 'Zero Padding was unsuccessful'
        self.NumberTransportBlocks = int(FinalNumberOfInputBits/self.TransportBlockSize)

        # Allocate memory for the output vector
        NumberOfOutputBits         = self.NumberTransportBlocks * self.OutputBlockSize
        OutputBits                 = np.zeros([1, NumberOfOutputBits], np.int8)

        # Encode the Input Bits
        # A place holder for the transport block
        TransportBlock      = np.zeros([1, self.TransportBlockSize], dtype = ZeroPaddedInputBits[0].dtype)
        for BlockIndex in range(0, self.NumberTransportBlocks):
            InputStartIndex     = BlockIndex * self.TransportBlockSize
            TransportBlock[0,:] = ZeroPaddedInputBits[InputStartIndex:InputStartIndex + self.TransportBlockSize]
            OutputBitBlock      = np.mod(np.matmul(TransportBlock, self.G), 2)

            OutputStartIndex                                                        = BlockIndex * self.OutputBlockSize
            OutputBits[0, OutputStartIndex:OutputStartIndex + self.OutputBlockSize] = OutputBitBlock


        if strSoftbitMapping == 'hard':
            return OutputBits[0]
        elif strSoftbitMapping == 'inverting':         # Here we map bit 0 to +1 and bit 1 to -1
            InvertedOutputBits = -(2*OutputBits[0] - 1)
            return InvertedOutputBits
        else:                                          # Here we map bit 0 to -1 and bit 1 to +1
            NonInvertedOutputBits = 2*OutputBits[0] - 1 
            return NonInvertedOutputBits





    # ----------------------------------------------------------------------------
    # > This is the LDPC Message Passing Decoder procedure
    # ----------------------------------------------------------------------------
    def DecodeBits(self
                 , InputBeliefs:      np.ndarray
                 , NumberIterations:  int = 8
                 , strSoftbitMapping: str = 'non-inverting') -> np.ndarray:
        
        '''
        brief: This is the LDPC Message Passing Decoder
        param: InputBeliefs      -  A vector of LLR bit beliefs
        param: NumberIteration   -  Self explanatory
        param: strSoftbitMapping -  Softbit to hardbit mapping ('non-inverting' = -1/1 maps to 0/1   --- 'inverting' = -1/1 maps to 1/0)
                                    IEEE - is 'non-inverting', 3GPP - is 'inverting'
        '''
        # --------------------------------------
        # Type checking 
        assert isinstance(InputBeliefs, np.ndarray),           'The InputBeliefs input argument must be of type np.ndarray'
        assert np.issubdtype(InputBeliefs.dtype, np.floating), 'The InputBeliefs input argument must be an array of floating point values'
        assert isinstance(NumberIterations, int),              'The NumberIterations input argument must be an integer.'
        assert isinstance(strSoftbitMapping, str),             'The "strSoftbitMapping" input argument must be of type str'
        
        # --------------------------------------
        # Error checking
        assert strSoftbitMapping == 'non-inverting' or strSoftbitMapping == 'inverting', 'The strSoftbitMapping input argument is invalid'
        assert len(InputBeliefs) % self.NumEncodedBits == 0,   'The number of input bits must be an integer multiple of self.NumEncodedBits'

        # --------------------------------------
        # Copy and reformat the input beliefs if necessary
        if strSoftbitMapping == 'inverting':
            IntrinsicBeliefs = -InputBeliefs.copy()
        else:
            IntrinsicBeliefs =  InputBeliefs.copy()

        # ---------------------------------------
        # Find the number of LDPC decoding operations are necessary for the entire input belief vector
        NumLdpcRepetitions = int(len(InputBeliefs) / self.NumEncodedBits)
        RxBitEstimates     = np.zeros(NumLdpcRepetitions * self.NumMessageBits, np.uint8)

        # Iterate through each Ldpc Process
        for Repetition in range(0, NumLdpcRepetitions):
            StartIndexBeliefs   = Repetition*self.NumEncodedBits
            StartIndexMessage   = Repetition*self.NumMessageBits
            IntrinsicBeliefTemp = IntrinsicBeliefs[StartIndexBeliefs:StartIndexBeliefs + self.NumEncodedBits]
            CurrentBeliefs      = IntrinsicBeliefTemp

            E                = np.zeros(self.H.shape, InputBeliefs.dtype)
            for Interation in range(0, NumberIterations):
                # --------------------------
                # Step 1 and 4
                M = np.ones(self.H.shape, InputBeliefs.dtype)
                for row in range(0, self.H.shape[0]):
                    M[row, :] *= CurrentBeliefs

                M = M * self.H                 # Zero out positions that are not interesting to us
                                               # This helps when we add values column wise to find the total extrinsic belief
                M = M - E                      # During the first pass E == 0 (Step 1)
                                               # During later passes,  E != 0 (Step 4)

                # --------------------------
                # Step 2 and 5
                for CheckNodeIndex in range(0, self.L):
                    # Find the indices of non-zero entries in row 'CheckNodeIndex' of the parity check matrix H
                    CheckNodeConnectionIndices = np.nonzero(self.H[CheckNodeIndex,:])
                    # Fetch the extrinsic beliefs at that row and send to SISO_SPC_Decoder
                    rn     = M[CheckNodeIndex, CheckNodeConnectionIndices]
                    l      = CLdpcProcessor.SISO_SPC_Decoder(rn[0])
                    # This matrix remembers the intrinsic beliefs that we need to subtract later
                    E[CheckNodeIndex, CheckNodeConnectionIndices] = l
                

                # Step 4:
                # Find the sum of the extrinsic beliefs for a particular bit node.
                TotalExtrinsicBeliefs = np.sum(E, axis=0)
       
                # Add that sum to the intrinsic belief (original received bit beliefs) to get
                # the new updated intrinsic belief = r[x]new in the text.
                CurrentBeliefs      = IntrinsicBeliefTemp + TotalExtrinsicBeliefs

            # Map the received beliefs back to bits
            RxBitEstimates[StartIndexMessage:StartIndexMessage + self.NumMessageBits] = 0.5 * (np.sign(CurrentBeliefs[0:self.K]) + 1)

        # Recast to a reasonable type
        RxBitEstimates = RxBitEstimates.astype(np.uint8)
        return RxBitEstimates














# --------------------------------------------------------------------------------
# > Test bench
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    # ---------------------------------
    # Verifying the CRC Computation
    # ---------------------------------
    # -> The following test vector is also used in the MatLab code: GenerateCRC_tb.m
    Message = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, \
                        1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, \
                        1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    
    # -------------------------------------------------------------------------------------------
    # Compute and check the 16 bit CRC. The correct sequence below comes from the MatLab test bench
    CorrectCrcOutput16 = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0]
    CrcOutput16 = CCrcProcessor.ComputeCrc(16, Message)
    Errors      = np.sum(np.remainder(CorrectCrcOutput16 + CrcOutput16, 2))
    assert Errors == 0, 'The 16 bit CRC test failed.'

    # -------------------------------------------------------------------------------------------
    # Compute and check the 24 bit CRC. The correct sequence below comes from the MatLab test bench
    CorrectCrcOutput24 = [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    CrcOutput24 = CCrcProcessor.ComputeCrc(24, Message)
    Errors      = np.sum(np.remainder(CorrectCrcOutput24 + CrcOutput24, 2))
    assert Errors == 0, 'The 24 bit CRC test failed.'





    # ------------------------------------------- 
    # Verifying the LDPC Processor
    # ------------------------------------------- 
    Test = 6           # 0 - Test TransformParityCheckMatrix() and ComputeGeneratorMatrix() using some small
                       #     test example matrices 
                       # 1 - Transform a Prototype matrix into a ParityCheckMatrix using CreateParityCheckMatrix()
                       # 2 - Test all three functions using all Prototype Matrices. 
                       # 3 - Testing LDPC encoding
                       # 4 - Testing the SISO_SPC_Decoder
                       # 5 - Testing the message passing decoder with the book example
                       # 6 - Testing all WLAN variant of the LDPC processor


    LdpcProcessor = CLdpcProcessor()
    # ------------
    if Test == 0:
        # -> Here we test the Ldpc functionality on some very simple matrices
        #    We will check the TransformParityCheckMatrix() function and the
        #    ComputeGeneratorMatrix() function.
        H = np.array([[1, 1, 1, 1, 0, 0],
                      [1, 0, 0, 1, 1, 0],
                      [0, 0, 1, 1, 0, 1]], dtype = np.int8)

        H = np.array([[1, 1, 1, 1, 0, 1, 0],
                      [1, 0, 0, 1, 1, 0, 1], 
                      [0, 0, 1, 1, 0, 1, 1],
                      [1, 0, 0, 1, 0, 0, 1]], dtype = np.int8)
    
        H_New = CLdpcProcessor.TransformParityCheckMatrix(H)

        # Use the following function to show the matrix in Notepad++
        DebugUtility.ShowMatrix(H_New)

        # The generator matrix is verified internally and would assert upon failure
        G     = CLdpcProcessor.ComputeGeneratorMatrix(H)

        

    # ------------
    if Test == 1:
        H = LdpcProcessor.CreateParityCheckMatrix(648, '3/4')
        DebugUtility.ShowMatrix(CLdpcProcessor.PrototypeM648_3_4, 'PrototypeM648_3_4.txt')
        DebugUtility.ShowMatrix(H)

    # -----------
    if Test == 2:
        print('This test takes several seconds.')
        H = LdpcProcessor.CreateParityCheckMatrix(648, '1/2')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H) 

        H = LdpcProcessor.CreateParityCheckMatrix(648, '2/3')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H)

        H = LdpcProcessor.CreateParityCheckMatrix(648, '3/4')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H)

        H = LdpcProcessor.CreateParityCheckMatrix(648, '5/6')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H)

        H = LdpcProcessor.CreateParityCheckMatrix(1296, '1/2')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H) 

        H = LdpcProcessor.CreateParityCheckMatrix(1296, '2/3')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H)

        H = LdpcProcessor.CreateParityCheckMatrix(1296, '3/4')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H)

        H = LdpcProcessor.CreateParityCheckMatrix(1296, '5/6')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H)

        H = LdpcProcessor.CreateParityCheckMatrix(1944, '1/2')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H) 

        H = LdpcProcessor.CreateParityCheckMatrix(1944, '2/3')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H)

        H = LdpcProcessor.CreateParityCheckMatrix(1944, '3/4')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H)

        H = LdpcProcessor.CreateParityCheckMatrix(1944, '5/6')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H)
    
        print('The test has passed.')


    # ------------
    if Test == 3:
        H = LdpcProcessor.CreateParityCheckMatrix(648, '1/2')
        G = CLdpcProcessor.ComputeGeneratorMatrix(H) 

        NumBits = 324
        InputBits = np.random.randint(low=0, high = 2, size = (1, NumBits), dtype = np.uint8)
        EncodedBits = LdpcProcessor.EncodeBits(InputBits, 'hard', True)
        Stop = 1

    # ------------
    if Test == 4:
        r = np.array([-0.1, -1, -0.9, 1.2], dtype = np.float32)
        print(CLdpcProcessor.SISO_SPC_Decoder(r))


    # -----------
    if Test == 5:
        # Test repeats the book example in section 5.6.5.3 of the book
        H = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],\
                      [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],\
                      [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],\
                      [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],\
                      [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],\
                      [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],\
                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]], dtype = np.int8)

        LdpcProcessor1 = CLdpcProcessor('custom', H)

        # TxBipolarBits = [ 1  -1   1   -1  1  -1   1     -1   -1    1  -1    1   1  -1
        # Define Rx bit beliefs forcing errors are in positions 2, 7, 12
        RxBeliefs  = np.array([2, .1,  1.5,  -1,  1,  -1,  -0.2,  -0.8,  -0.8,  1, -1,  -0.3,  1,  -1], np.float32)

        # Run the message passing decoder
        OutputBits = LdpcProcessor1.DecodeBits(RxBeliefs 
                                             , NumberIterations = 10
                                             , strSoftbitMapping = 'non-inverting') 

        ProperAnswer = np.array([1, 0, 1, 0, 1, 0, 1], np.uint8)
        assert all(OutputBits == ProperAnswer), 'Test 5 has failed'
        print('Test 5 has passed.')


    # -----------
    if Test == 6:
        # Parameter for bit encoding
        NumInputBits      = 324 * 100
        strBitMode        = 'non-inverting'
        bAllowZeroPadding = False
        InputBits         = np.random.randint(low = 0, high = 2, size = NumInputBits, dtype = np.uint8)

        SnrdB_List         = [0, 1, 2, 3, 4]
        NumEncodedBitsList = [648, 1296, 1944] 
        RateList           = ['1/2', '2/3', '3/4', '5/6']
        BER_List           = []

        # Iterate through the LDPC sizes
        BerListIndex       = 0
        for NumEncodedBits in NumEncodedBitsList:
            # Iterate through the rate
            for Rate in RateList:
                # Increment list index
                BER_List.append([0] * len(SnrdB_List))
                
                # Run the test for Number of Encoded bit = 648 and rate 1/2
                print('Building LdpcProcess for ' + str(NumEncodedBits) + ' bits at rate ' + Rate + '.')
                LdpcProcessor1 = CLdpcProcessor('WLAN', None, 648, '1/2')
                EncodedBits = LdpcProcessor1.EncodeBits(InputBits
                                                      , strBitMode                
                                                      , bAllowZeroPadding)

                # Iterate through the SINR
                for SnrIndex, SnrdB in enumerate(SnrdB_List):
                    print('Decoding at Snr = ' + str(SnrdB) + 'dB.')
                    RxBeliefs = sp.AddAwgn(SnrdB, EncodedBits)

                    # Run the message passing decoder
                    OutputBits = LdpcProcessor1.DecodeBits(RxBeliefs 
                                                         , NumberIterations = 10
                                                         , strSoftbitMapping = 'non-inverting') 

                    # Compute the BER
                    NumErrors = np.sum(np.mod(InputBits + OutputBits, 2))
                    BER       = NumErrors / NumInputBits
                    BER_List[BerListIndex][SnrIndex] = BER
                    
                    print('The BER = ' + str(BER))

                BerListIndex += 1

    plt.figure(1)
    plt.subplot(4,1,1)
    plt.semilogy(SnrdB_List, BER_List[0], 'r')
    plt.semilogy(SnrdB_List, BER_List[1], 'b')
    plt.semilogy(SnrdB_List, BER_List[2], 'k')
    plt.semilogy(SnrdB_List, BER_List[3], 'k:')
    plt.grid(True)
    plt.tight_layout()
    plt.title('BER for Encoded Bit Size = 648')
    plt.xlabel('SNR')
    plt.xlabel('BER')
    plt.legend(['1/2', '2/3', '3/4', '5/6'])
    plt.subplot(4,1,2)
    plt.semilogy(SnrdB_List, BER_List[4], 'r')
    plt.semilogy(SnrdB_List, BER_List[5], 'b')
    plt.semilogy(SnrdB_List, BER_List[6], 'k')
    plt.semilogy(SnrdB_List, BER_List[7], 'k:')
    plt.grid(True)
    plt.tight_layout()
    plt.title('BER for Encoded Bit Size = 1296')
    plt.xlabel('SNR')
    plt.xlabel('BER')
    plt.legend(['1/2', '2/3', '3/4', '5/6'])
    plt.subplot(4,1,3)
    plt.semilogy(SnrdB_List, BER_List[8], 'r')
    plt.semilogy(SnrdB_List, BER_List[9], 'b')
    plt.semilogy(SnrdB_List, BER_List[10], 'k')
    plt.semilogy(SnrdB_List, BER_List[11], 'k:')
    plt.grid(True)
    plt.tight_layout()
    plt.title('BER for Encoded Bit Size = 1944')
    plt.xlabel('SNR')
    plt.xlabel('BER')
    plt.legend(['1/2', '2/3', '3/4', '5/6'])
    plt.show()
