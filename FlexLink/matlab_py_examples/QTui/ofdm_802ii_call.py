import numpy as np
# from scipy.signal.windows import kaiser
# from scipy.signal import sinc
from numpy import kaiser
from numpy import sinc
import matplotlib.pyplot as plt
import Visualization

vis = Visualization.Visualization()
debug = False

def ofdm_802ii_call(d_frq, d_tm, d_clk, d_gn, d_phs, del_t):
    """
    d_frq : adjusts the carrier frequency difference this value should be from -1 to 1
    d_tm  : adjusts the frame time the range should be -1 to 1
    d_clk : adjusts the sampling clock offset range is -1 to 1
    d_gn  : adjusts the IQ gain imbalance range from -0.2 to 0.2
    d_phs : adjusts the IQ phase imbalance range from -0.2 to 0.2
    del_t : adjusts the IQ differential time range from -0.2 to 0.2
    """
    fft_size = 128
    f1dat = np.zeros(fft_size, dtype=complex)

    # Generating the range vector adjusted by del_t
    # In Python, np.arange does not include the endpoint, similar to Matlab's colon operator when used with integer steps.
    t = np.arange(-4 + del_t, 4 + del_t + 1)  # +1 to include the endpoint

    # Calculate the sinc function
    # The scipy.signal.sinc function expects input x as pi*x, differently from MATLAB's sinc which is normalized by pi.
    sinc_values = sinc(t * np.pi)

    # Generate the Kaiser window
    # Beta is 5 and the number of points is 9
    kaiser_window = kaiser(9, 5)

    # Element-wise multiplication
    hh1 = sinc_values * kaiser_window
    
    # Define filter kernels
    # same as above: hh1 = sinc(np.arange(-4 + del_t, 5 + del_t)) * kaiser(9, 5)
    hh1 /= np.sum(hh1)
    hh2 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])

    hh1a = sinc(np.arange(-4 + d_tm, 5 + d_tm)) * kaiser(9, 5)
    hh1a = hh1a / np.sum(hh1a)

    hh1b = sinc(np.arange(-4.1, 5.0 - 0.1)) * kaiser(9, 5)
    hh1c = sinc(np.arange(-3.9, 5.0 + 0.1)) * kaiser(9, 5)
    hh1d = 5 * (hh1b - hh1c)

    fdet2 = np.zeros((30, 57), dtype=complex)  # Prepare array for results
    data_x = np.loadtxt('data_x.txt')
    data_y = np.loadtxt('data_y.txt')

    print(data_x.shape, data_y.shape) # should be 30 x 
    dx1 = (np.floor(4 * data_x) - 1.5) / 1.5
    dy1 = (np.floor(4 * data_y) - 1.5) / 1.5
    dx1[26], dy1[26] = 0, 0  # Zeroing the center elements
    
    for mm in range(30):
        fd1 = np.zeros(fft_size, dtype=complex)
        # dx1 = (np.floor(4 * np.random.rand(52)) - 1.5) / 1.5
        # dy1 = (np.floor(4 * np.random.rand(52)) - 1.5) / 1.5

        fd1[64-26:64+26] = dx1[mm,:] + 1j * dy1[mm,:]
        fd1 = np.fft.fftshift(fd1)
        d1 = np.fft.ifft(fd1)    # in time domain

    
        if debug == True: 
            vis.plot_data([np.real(fd1),np.real(d1)], names=['fd1','d1'], start=0, points=False)
            vis.plot_constellation(fd1,name='fd1')
            vis.plot_constellation(d1,name='d1')


        # Apply filters and delays
        # delay quadrature response relative to in-phase response
        x1 = np.real(d1)
        y1 = np.imag(d1)
        # add cyclic prefix and then delay quad signal and strip prefix
        x2 = np.concatenate([x1[-9:], x1, x1[:9]])
        y2 = np.concatenate([y1[-9:], y1, y1[:9]])

        x3 = np.convolve(x2, hh2, mode='full')
        y3 = np.convolve(y2, hh1, mode='full')
        
        # was: x4 = np.convolve(np.concatenate([x1[-9:], x1, x1[:9]]), hh2, mode='full')[13:141]
        x4 = x3[13:141]
        y4 = y3[13:141]
        # differential delay on I and Q has been inserted

        
        d1a = x4 + 1j * (1 + d_gn) * y4 * np.exp(1j * d_phs)
        # gain and phase imbalance inserted
        if debug == True: 
            vis.plot_data([np.real(d1a)], ['d1a'], start=0, points=False)
            vis.plot_constellation(d1a,name='d1a')
        
        
        # adding frequency offset
        d1a = d1a * np.exp(-1j * 2 * np.pi * np.arange(-64, 64) * d_frq / 128)
        # shifted spectrum
        if debug == True:
            vis.plot_data([np.real(d1a)], ['d1a'], start=0, points=False)
            vis.plot_constellation(d1a,name='d1a')
            
        #"""
        #----------------------------------------------------------------------
        # now insert block delay
        x1 = np.real(d1a)
        y1 = np.imag(d1a)
        # add cyclic prefix and then delay quad signal and strip prefix
        x2 = np.concatenate([x1[-9:], x1, x1[:9]])
        y2 = np.concatenate([y1[-9:], y1, y1[:9]])
        
        x3 = np.convolve(x2, hh1a, mode='full')
        y3 = np.convolve(y2, hh1a, mode='full')

        x4 = x3[13:141]
        y4 = y3[13:141]
        
        d1a = x4 + 1j * y4
        # block delay on I and Q has been inserted
        if debug == True:
            vis.plot_data([np.real(d1a)], ['d1a'], start=0, points=False)
            vis.plot_constellation(d1a,name='d1a')
        #----------------------------------------------------------------------
        
        #""" 
        #----------------------------------------------------------------------
        #% now insert clock offset
        x1 = np.real(d1a)
        y1 = np.imag(d1a)
        # add cyclic prefix and then delay quad signal and strip prefix
        x2 = np.concatenate([x1[-9:], x1, x1[:9]])
        y2 = np.concatenate([y1[-9:], y1, y1[:9]])

        x3 =np.convolve(x2,hh1d, mode='full')
        y3 =np.convolve(y2,hh1d, mode='full')
                
        x3a=np.convolve(x2,hh2, mode='full')
        y3a=np.convolve(y2,hh2, mode='full')

        x3b=x3a+(d_clk/153)*np.arange(-77,77)*x3
        y3b=y3a+(d_clk/153)*np.arange(-77,77)*y3
        # --------------------------------------
        
        # x3b = np.convolve(x2, hh1d, mode='full') + (d_clk / 153) * np.arange(-77, 77) * np.convolve(x2, hh2, mode='full')
        # y3b = np.convolve(y2, hh1d, mode='full') + (d_clk / 153) * np.arange(-77, 77) * np.convolve(y2, hh2, mode='full')
        x4= x3b[13:141]
        y4= y3b[13:141]
        
        # d1a = x3b[13:141] + 1j * y3b[13:141]
        d1a = x4 + 1j * y4
        # block delay on I and Q has been inserted
        if debug == True:
            vis.plot_data([np.real(d1a)], ['d1a'], start=0, points=False)
            vis.plot_constellation(d1a,name='d1a')
        #----------------------------------------------------------------------
        #"""
        # clock offset inserted
        # insert clock offset ??
        fd2a = np.fft.fftshift(np.fft.fft(d1a))
        
        if debug == True: 
            vis.plot_data([np.real(fd2a),np.imag(fd2a) ], ['real.fd2a', 'imag.fd2a'], start=0, points=False)
            vis.plot_constellation(fd2a,name='d1a')
        
        # fdet2[mm, :] = fd2a[64 - 28:64 + 29]
        # fdet2[mm, :] = fd2a[64 - 28:64 + 29]
        # list comprehension = [x for x in fd2a if x ]
        fd2a_temp   = fd2a[(64-28):(64+28+1)]
        fdet2[mm, :] = fd2a_temp[::-1]
        # fdet2[mm, :] = fd2a[(64+28):(64-28+1)]
        
        # fdet2.append(fd2a[64 - 29:64 + 29])
    if debug == True: 
        for i in range(fdet2.shape[0]):
            print(f"row {i}:", fdet2[i,:])
        
    return fdet2

# Example of using this function
# fdet2 = ofdm_802ii_call(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
