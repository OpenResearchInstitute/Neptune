"""    
    Parameters Setup:
        It initializes constants like the sample rate, FFT size, cyclic prefix 
        length, and calculates the total OFDM symbol length and subcarrier spacing.
        It converts 1 microsecond into sample units for phase compensation purposes.

    Error Checking:
        Ensures the starting sample index is properly offset to avoid intersymbol 
        interference from channel precursors.

    OFDM Configuration:
        Sets the number of subcarriers and the indices for both positive and negative 
        subcarrier mappings depending on whether it's using LTE or WLAN bandwidth configurations.

    Phase Compensation:
        Calculates a phase compensation vector to adjust for the timing advance of 
        1 microsecond, accounting for frequency offsets.

    Demodulation Process:
        Initializes the resource grid.
        Iteratively processes each OFDM symbol in the input sequence:
            Extracts the FFT input buffer.
            Performs an FFT.
            Maps the FFT output to the resource grid, applying phase compensation.
            
            
possible improvements:

    Vectorization and Optimization: The Python version benefits from NumPy's efficient 
    array operations, ensuring that the entire operation is vectorized where possible, 
    which is generally faster and more efficient than loop-based equivalents in MATLAB.

    Error Handling: Python uses assertions similar to MATLAB for basic input validation, 
    but in a production environment, it might be better to raise exceptions with descriptive 
    messages.

    Phase Compensation: The calculation of the phase compensation and its application are 
    directly translated and make use of NumPy's broadcasting capabilities, which simplifies 
    the implementation.

    Subcarrier Mapping: The code explicitly handles the mapping of FFT indices to the 
    resource grid, maintaining clarity and correctness in translating frequency domain 
    data back to its subcarrier positions.
"""



import numpy as np

def ofdm_demodulator(input_sequence, start_sample, b_lte_bw):
    sample_rate = 20e6  # 20MHz
    fft_size = 1024
    cp_length = 116
    ofdm_symbol_length = cp_length + fft_size
    subcarrier_spacing = sample_rate / fft_size


    """ An OFDM symbol is the concatenation of the CP and the IFFT portion
    The StartSample input argument is the estimated position of the first sample
    of the Ifft portion of the first valid OFDM symbol in the FlexLink Packet.
    The start time is found via the correlation against PreambleB.
    We will start 1 microsecond inside the CP to avoid intersymbol 
    interference due to channel precusors."""
    one_microseconds_in_samples = 20
    assert start_sample > one_microseconds_in_samples, "Improper input sequence."

    # Determine the number of available OFDM symbols we can demodulator
    num_available_ofdm_symbols = len(input_sequence) // ofdm_symbol_length

    if b_lte_bw:  # LTE BW
        num_subcarriers = 913
        pos_subcarrier_indices = np.arange(456, 913)
        neg_subcarrier_indices = np.arange(456)
        pos_fft_indices = np.arange(457)
        neg_fft_indices = np.arange(568, 1024)
    else:  # WLAN BW
        num_subcarriers = 841
        pos_subcarrier_indices = np.arange(420, 841)
        neg_subcarrier_indices = np.arange(420)
        pos_fft_indices = np.arange(421)
        neg_fft_indices = np.arange(604, 1024)

    # Phase compensation for 1 microsecond advance
    tones_ieee = np.arange(-(num_subcarriers - 1)//2, (num_subcarriers - 1)//2 + 1)
    one_microsecond = one_microseconds_in_samples / sample_rate
    compensation = np.exp(1j * 2 * np.pi * one_microsecond * tones_ieee * subcarrier_spacing)


    #output_waveform = waveform*np.exp(1j*2*np.pi*(delta_f/Fs)*np.arange(len(waveform)) +
    #                                  1j * phaseoff)


    # Start the OFDM Demodulation Process
    resource_grid = np.zeros((num_subcarriers, num_available_ofdm_symbols), dtype=complex)
    start_index = start_sample - one_microseconds_in_samples
    range_indices = np.arange(start_index, start_index + fft_size)

    for idx in range(num_available_ofdm_symbols-1):
        fft_input_buffer = input_sequence[range_indices]
        fft_output_buffer = np.fft.fft(fft_input_buffer)

        # Place the compensated FFT output into the resource grid
        resource_grid[pos_subcarrier_indices, idx] = fft_output_buffer[pos_fft_indices]
        resource_grid[neg_subcarrier_indices, idx] = fft_output_buffer[neg_fft_indices]
        
        ### resource_grid[:, idx] *= compensation

        range_indices += ofdm_symbol_length

    return resource_grid
