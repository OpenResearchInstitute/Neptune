# File:     DebugUtility.py
# Notes:    This template includes basic debugging functions that help in the development of the FlexLink Phy

__title__     = "DebugUtility"
__author__    = "Andreas Schwarzinger"
__status__    = "preliminary"
__date__      = "Sept, 10rd, 2022"
__copyright__ = 'Andreas Schwarzinger'


import numpy as np
import os


# -------------------------------------------------------------------
# > ShowMatrix() 
# -------------------------------------------------------------------
def ShowMatrix(InputMatrix:  np.ndarray
             , strFileName:  str  = 'Matrix.txt'
             , bShow:        bool = True) -> str:
    """
    notes: The show matrix function will write a 2 dimensional numpy array to a file for easy visualization.
    param: InputMatrix  I   This must be a numpy array
    param: strFileName  I   The matrix will be formatted and writted to a file with name FileName
    param: bShow        I   This function can show the file content in notepad
    """

    # -------------------------------------------
    # Check errors
    assert isinstance(InputMatrix, np.ndarray), 'The InputMatrix argument must be of type numpy.ndarray'
    assert InputMatrix.ndim == 2,               'The InputMatrix must feature two dimensions.'
    assert isinstance(strFileName, str),        'The strFileName argument must be of type str'
    assert isinstance(bShow, bool),             'The bShow argument must be of type bool'

    bIsInteger = np.issubdtype(InputMatrix.dtype, np.integer)
    bIsFloat   = np.issubdtype(InputMatrix.dtype, np.floating)
    bIsComplex = np.issubdtype(InputMatrix.dtype, np.complex64) or np.issubdtype(InputMatrix.dtype, np.complex128)

    assert bIsInteger or bIsFloat or bIsComplex, 'The dtype of the input matrix is unsupported'

    # ---------------------------------------------
    # Write the matrix to a string so we may print it out
    rows, columns = InputMatrix.shape
    strMatrix = ''

    for row in range(0, rows + 1):
        for column in range(0, columns + 1):
            if row == 0 and column == 0:
                strMatrix += '       | '
                continue

            if column == 0:
                strMatrix += "{:6d}".format(row -1) + ' | '

            if row == 0:
                if bIsInteger:
                    strMatrix += '{:^5}'.format(column - 1) + ' | '
                if bIsFloat:
                    strMatrix += '{:^15}'.format(column - 1) + ' | '
                if bIsComplex:
                    strMatrix += '{:^22}'.format(column - 1) + ' | '

            if row > 0 and column > 0 :
                if bIsInteger:
                    strMatrix += '{:^5}'.format(InputMatrix[row-1, column-1]) + ' | '
                if bIsFloat:
                    strMatrix += '{:^15.7e}'.format(InputMatrix[row-1, column-1]) + ' | '
                if bIsComplex:
                    strMatrix += '{:^20.4e}'.format(InputMatrix[row-1, column-1]) + ' | '        

            if column == columns:
                strMatrix += ' \n'
        

    # Show the matrix in notepad if desired
    if bShow == True: 
        # -------------------
        f = open(file = strFileName, mode = 'w', encoding = 'utf-8')
        f.write(strMatrix)
        f.close()
        os.system("notepad++ " + strFileName)
    else:
        return strMatrix




# -------------------------------------------------------
# Testbench
if __name__ == '__main__':
    A = np.zeros([2, 4], dtype = np.int8)
    A[1,1] = 1 
    A[0,0] = 2
    ShowMatrix(A)

    Done = 1

