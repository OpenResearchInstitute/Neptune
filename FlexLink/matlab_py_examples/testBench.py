"""    
Environment Setup:
    Clears variables, closes all figures, and clears the command window, setting 
    up a clean environment for the test.

Configuration:
    Sets the number of OFDM symbols and the transmission sample rate.
    Configures the signal-to-noise ratio (SNR) to make the plots more interesting, 
    indicating a relatively high-quality signal.
    Determines the number of subcarriers based on whether LTE bandwidth settings 
    are used.

Resource Grid Creation:
    Generates a transmit (Tx) resource grid using QPSK modulation. Real (I) 
    and imaginary (Q) components are derived from binary values, scaled to ensure unit energy per symbol.

OFDM Modulation:
    Passes the resource grid and the transmission sample rate to an OFDM 
    modulator function, which presumably converts the grid into a time-domain OFDM sequence.

Noise Addition:
    Adds Gaussian noise to the modulated sequence. The noise power is calculated 
    based on the defined SNR, and noise is generated separately for the I and Q 
    components, then combined to form complex-valued noise.
    Measures the actual noise power and the resulting SNR, displaying the 
    measured SNR in decibels.

Signal Reception:
    If the sample rate was initially set to 40 MHz, it decimates the signal 
    by 2 (downsampling by a factor of 2) and adjusts the power.

OFDM Demodulation:
    Demodulates the received, noisy sequence starting from a specific sample, 
    using an OFDM demodulator function. This simulates the receiver side of an 
    OFDM system.

Error Calculation:
    Calculates the difference between the transmitted and received resource 
    grids, finding the maximum absolute error to assess the system's performance.

Visualization:
    Plots the constellation diagrams for both the transmitted and received 
    resource grids to visually inspect the modulation quality and effects of noise.
"""


import numpy as np
import matplotlib.pyplot as plt
import ofdmDemodulator
from ofdmModulator import ofdm_modulator
from ofdmDemodulator import ofdm_demodulator



def plot_iq_data2x(data1, data2, name1, name2):
    """ 
    function plots 2 rows of data, 1 for real, and 1 for imaginary
    data1,2 : complex data to plot
    """

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(np.real(data1), 'b', label='real')
    ax1.plot(np.real(data1), 'r.')
    ax1.plot(np.imag(data1), 'g', label='imag')
    ax1.plot(np.imag(data1), 'r.')
    ax1.grid('on')
    ax1.legend()
    ax2.plot(np.real(data2), 'b', label='real')
    ax2.plot(np.real(data2), 'r.')
    ax2.plot(np.imag(data2), 'g', label='imag')
    ax2.plot(np.imag(data2), 'r.')
    ax2.grid('on')
    ax2.legend()
    ax1.set_ylabel(name1)
    ax2.set_ylabel(name2)
    # ax1.set_xlim([-Tb, 4 * Tb])
    # ax2.set_xlim([-Tb, 4 * Tb])
    plt.legend()
    plt.show(block=False)

def plot_iiqq_data2x(data1, data2, name1, name2):
    """ 
    function plots compares the real values of 2 data sets and 2
    values of imaginary
    data1,2 : complex data to plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(np.real(data1), 'b', label='data 1 real')
    ax1.plot(np.real(data1), 'r.')
    ax1.plot(np.real(data2), 'g', label='data 2 real')
    ax1.plot(np.real(data2), 'r.')
    ax1.grid('on')
    ax1.legend()
    ax2.plot(np.imag(data1), 'b', label='data 1 imag')
    ax2.plot(np.imag(data1), 'r.')
    ax2.plot(np.imag(data2), 'g', label='data 2 imag')
    ax2.plot(np.imag(data2), 'r.')
    ax2.grid('on')
    ax2.legend()
    ax1.set_ylabel(name1)
    ax2.set_ylabel(name2)
    # ax1.set_xlim([-Tb, 4 * Tb])
    # ax2.set_xlim([-Tb, 4 * Tb])
    plt.legend()
    plt.show(block=False)



# Configuration
num_ofdm_symbols = 2
tx_sample_rate = 20e6
snr_db = 20
b_lte_bw = True
num_subcarriers = 913 if b_lte_bw else 841

# Create the Transmit Resource Grid (all QPSK)
i = (np.sqrt(2)/2) * (np.random.randint(0, 2, (num_subcarriers, num_ofdm_symbols)) - 0.5)
q = (np.sqrt(2)/2) * (np.random.randint(0, 2, (num_subcarriers, num_ofdm_symbols)) - 0.5)
tx_resource_grid = i + 1j*q

# Run the FlexLink OFDM Modulator
tx_output_sequence = ofdm_modulator(tx_resource_grid, tx_sample_rate)  # Assuming implementation

# Add some noise
snr_linear = 10**(snr_db/10)
signal_power = np.mean(np.abs(tx_output_sequence)**2)
noise_power = signal_power / snr_linear
i_noise = np.sqrt(0.7071 * noise_power) * np.random.randn(*tx_output_sequence.shape)
q_noise = np.sqrt(0.7071 * noise_power) * np.random.randn(*tx_output_sequence.shape)
noise = i_noise + 1j*q_noise
meas_noise_power = np.mean(np.abs(noise)**2)
meas_snrdB = 10 * np.log10(signal_power / meas_noise_power)
print(f'Measured SNR (dB): {meas_snrdB}')

rx_input_sequence = tx_output_sequence + noise

# Decimate to 20MHz if we generated everything at 40MHz
if tx_sample_rate == 40e6:
    rx_input_sequence = 2 * rx_input_sequence[::2]

# Demodulate the received, noisy sequence
start_sample = 116
start_sample = 137

rx_resource_grid = ofdm_demodulator(rx_input_sequence, start_sample, b_lte_bw)  # Assuming implementation

# Error calculation
error = tx_resource_grid - rx_resource_grid
max_abs_error = np.max(np.abs(error))
print(f'Maximum Abs Error Magnitude = {max_abs_error}')


# Plotting
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.real(tx_resource_grid.ravel()), np.imag(tx_resource_grid.ravel()), 'b.')
plt.grid(True)
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Constellation of TX Resource Elements')

plt.subplot(1, 2, 2)
plt.plot(np.real(rx_resource_grid.ravel()), np.imag(rx_resource_grid.ravel()), 'b.')
plt.grid(True)
plt.xlabel('I')
plt.ylabel('Q')
plt.title('Constellation of RX Resource Elements')
plt.show()


plot_iiqq_data2x(tx_resource_grid[:,0], rx_resource_grid[:,0], "tx grid", "rx grid")


print("stop")

