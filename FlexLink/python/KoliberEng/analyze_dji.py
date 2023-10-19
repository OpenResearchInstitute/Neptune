## not finished, work in progress
## copyright: Leonard Dieguez                        Aug 2022


import os
import struct
# import sys
# import csv
# import json
from datetime import datetime
# from signal import pause
from socket import gethostname
# import preprocess_data
# import visualization as vis
import numpy as np
import pandas as pd
# import pickle
# import gc
import struct
import matplotlib.pyplot as plt
from scipy.signal import firwin, upfirdn, lfilter, freqz, hilbert
#https://scikit-dsp-comm.readthedocs.io
from sk_dsp_comm import digitalcom as dc


#local
import visualization
import comms_utils
import ModulateDemodulate


if __name__ =='__main__':

    USE_BIG_ENDIAN = False
    USE_SMALL_ENDIAN = not USE_BIG_ENDIAN

    today = datetime.now()
    current_date = today.strftime('%Y-%m-%d')
    current_datetime = today.strftime('%Y-%m-%d_%H%M')

    main_path = os.getcwd()
    data_folder = os.path.join(main_path, 'data')
    # file_name = 'with_rc_2_2450000.0kHz_200.0MHz.iq'
    file_name = 'signal.1.92M.dat'

    fpath = os.path.join(data_folder , file_name)
    fh = open(fpath, "rb")

    # iq_from_file =  np.fromfile(fpath, dtype=np.int16, count=2)

    # Fs = 200e6    # in Hz.
    Fs = 30.72e6
    Fs_lte = 30.72e6

    f_offset = 0*1024
    # number of samples at the LTE fft sample rate
    # see Moray, LTE and the Evolution to 4G wireless... ch2 p32 table 2.1-16
    # symbol 0 is 160, symbol 1 to 6 is 144. 
    # for LTE the cp and symbol is 160 + 2048 = 2192
    # or 144 + 2048 = 2192
    # cp to symbol ratio is 0.078125
    # from LTE-in-a-Nutshell-Physical-Layer. 
    # Tsubfr = 1ms, Tslot = 0.5 ms
    # one slot is ?7 symbols? 
    # note for 802.11a cp + symbol is: 16 + 64 = 80
    # cp to symbol ratio 0.25
    # NOTE: Ts = T + Tg ; Symbol time + Guard time (cp time)
    # Ts in samples for LTE is 
    n_symbols = 2
    n_fft = 2048
    n_cp =  144  # felix says the cp is 144
    n_Ts = n_fft + n_cp


    # number of samples at the actual sample rate needed 
    # prior to downsampling decimation
    n_valuesFs = int(np.ceil(n_symbols * n_Ts*Fs/Fs_lte))

    # generate the FFT frequecy bins for the lte sample rates
    freqs = np.linspace(Fs_lte/-2, Fs_lte/2, n_fft)
    freqsMHz = freqs/1000000

    iqvalues = np.zeros((n_valuesFs),dtype=complex)

    # should be even if looking at IQ values
    # Q: does this need to be a multiple of the read size?
    fh.seek(f_offset)
    fh.seek(0)
    if USE_BIG_ENDIAN:
        for i in range(n_valuesFs):
            # iq_tup = struct.unpack('HH', fh.read(4))
            # i_val = iq_tup[0]
            # q_val = iq_tup[1]
            # iq_tup = struct.unpack('BBBB', fh.read(4))
            # i_val = np.int16((iq_tup[1] << 8) + iq_tup[0])
            # q_val = np.int16((iq_tup[3] << 8) + iq_tup[2])

            iq_tup = struct.unpack('>ff', fh.read(8))
            i_val = iq_tup[0]
            q_val = iq_tup[1]

            # print(hex(i_val),hex(q_val))
            # print(i_val,q_val)
    iqvalues[i] = i_val + 1j*q_val

    if USE_SMALL_ENDIAN:
        for i in range(n_valuesFs):
            # iq_tup = struct.unpack('HH', fh.read(4))
            # i_val = iq_tup[0]
            # q_val = iq_tup[1]
            # iq_tup = struct.unpack('BBBB', fh.read(4))
            # i_val = np.int16((iq_tup[1] << 8) + iq_tup[0])
            # q_val = np.int16((iq_tup[3] << 8) + iq_tup[2])

            iq_tup = struct.unpack('<ff', fh.read(8))
            i_val = iq_tup[0]
            q_val = iq_tup[1]
            # print(hex(i_val),hex(q_val))
            # print(i_val,q_val)
            iqvalues[i] = i_val + 1j*q_val



    # downconvert the sampled symbols to baseband, remove freq offset. 
    x1 = comms_utils.add_frequency_offset(iqvalues, 200e6, -22.5e6, np.pi/2)  # phase offset in radians

    # create a halfband filter for decimation of input data from 200Msps 
    # to 31.72Msps N filter order of 32
    hb = firwin(33, 0.5)
    # threshold coeffients values below a level and set to zero
    hb[np.abs(hb) <= 1e-4] = 0.0
    #filter and decimate from 200 downto 50 
    x2 = upfirdn(h=hb, x=x1, down=4)
    # sk_dsp_comm.digitalcom.farrow_resample(x, fs_old, fs_new)
    # for LTE ofdm at 30.72 Msps the bandwidth is (1200*15KHz = 
    # 18MHz => 9MHz on either side of centre). That implies guard band 
    # of 1Mhz (from 20 MHz nominal bandwidth)
    x3 = dc.farrow_resample(x2, 50e6, 30.72e6)

    plt.figure()
    plt.plot(np.real(x3))
    # plt.plot(np.real(iqvalues))
    plt.ylabel('x3 values')
    plt.grid('on')
    plt.show(block=False)
    
    x3_fft = np.fft.fft(x3[-2048:])
    x3_fftShift = np.fft.fftshift(x3_fft)

    plt.figure()
    plt.plot(freqsMHz, 20*np.log10(np.abs(x3_fftShift)),'r.')
    # plt.plot(np.real(iqvalues))
    plt.ylabel('abs ffs values')
    plt.grid('on')
    plt.show(block=False)


    reg = np.zeros(n_Ts+1,dtype=complex)
    buf1 = np.zeros(n_cp, dtype=complex)
    buf2 = np.zeros(n_cp, dtype=complex)
    corr1 = np.zeros(n_Ts * n_symbols, dtype=complex)
    corr2 = np.zeros(n_Ts * n_symbols, dtype=complex)
    corr3 = np.zeros(n_Ts * n_symbols, dtype=complex)
    sqlsh = np.zeros(n_Ts * n_symbols, dtype=complex)
    # correlate over whole symbol time. 
    for n in range(n_symbols*n_Ts):
        reg = np.append(x3[n], reg[0:-1])
        prd1 = reg[0] * reg[n_fft].conj()
        prd2 = reg[n_fft] * reg[n_fft]
        buf1 = np.append(prd1, buf1[0:-1])
        buf2 = np.append(prd2, buf1[0:-1])
        corr1[n] = np.sum(buf1)/n_cp
        corr2[n] = np.sum(buf2)/n_cp
        corr3[n] = corr1[n]/(0.01 + corr2[n])
        sqlsh[n]=(np.sign(np.abs(corr3[n])-0.5) + 1) / 2


    plt.figure()
    plt.plot(np.abs(corr1))
    plt.plot(np.abs(corr2))
    plt.ylabel('abs corr 1,2')
    plt.grid('on')
    plt.show(block=False)
    
    plt.figure()
    plt.plot(np.abs(corr3))
    plt.plot(sqlsh)
    plt.ylabel('abs corr 1,2')
    plt.grid('on')
    plt.show(block=False)
    


    x1_fft = np.fft.fft(x1)
    x1_fftShift = np.fft.fftshift(x1_fft)

    plt.figure()
    plt.plot(freqsMHz, 20*np.log10(np.abs(x1_fftShift)),'r.')
    # plt.plot(np.real(iqvalues))
    plt.ylabel('abs ffs values')
    plt.grid('on')
    plt.show(block=False)

    plt.figure()
    plt.plot(np.real(x1),'r.')
    plt.plot(np.real(x1))
    plt.ylabel('imaginary values')
    plt.grid('on')
    plt.show(block=False)

    plt.figure()
    plt.plot(np.imag(x1),'r.')
    plt.plot(np.imag(x1))
    plt.ylabel('imaginary values')
    plt.grid('on')
    plt.show(block=False)


    plt.figure()
    plt.plot(np.real(x1), np.imag(x1),'r.')
    plt.plot(np.real(x1), np.imag(x1))
    plt.ylabel('IQ values')
    plt.grid('on')
    plt.show(block=False)


    file = open(fn, "rb")
    byte = file.read(4)


    while byte:

        print(byte)
        byte = file.read(4)


