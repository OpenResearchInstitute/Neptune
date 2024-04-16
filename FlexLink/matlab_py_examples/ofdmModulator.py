"""
        Input and Error Checking:
        The function accepts two inputs: ResourceGrid (a matrix where rows 
        correspond to subcarriers and columns to OFDM symbols) and OutputSampleRate.
        It checks that the number of subcarriers and the output sample rate are 
        valid for predefined configurations (LTE and WLAN bandwidths).

    IFFT and Cyclic Prefix Configuration:
        Based on the output sample rate, it sets the size of the Inverse Fast 
        Fourier Transform (IFFT) and the length of the cyclic prefix (CP). These 
        settings are crucial to match the timing requirements of the OFDM system.

    Subcarrier Mapping:
        Depending on the number of subcarriers, it identifies which subcarriers 
        are treated as positive and negative frequencies. It maps these 
        subcarriers to specific indices in the IFFT input buffer.

    OFDM Modulation Process:
        It initializes the output sequence.
        For each OFDM symbol, it populates the IFFT input buffer with 
        the resource grid data, performs an IFFT, and extracts a cyclic 
        prefix from the end of the IFFT result.
        It constructs each OFDM symbol by placing the cyclic prefix at 
        the beginning, followed by the IFFT result.
        This OFDM symbol is then placed into the output sequence.

    Output Sequence Assembly:
        The complete OFDM waveform is assembled by appending each symbol 
        sequentially into the OutputSequence.
        
            Subcarrier Indexing:
        The code defines positive and negative subcarrier indices based on 
        the number of subcarriers (913 for LTE and 841 for WLAN). It properly 
        segregates the positive and negative frequencies according to the resource grid size.
        The indices for pos_ifft_indices and neg_ifft_indices are used to map 
        the subcarriers from the resource grid into the IFFT input buffer. The 
        mapping is consistent with common OFDM practices where the DC subcarrier 
        (zero frequency) and negative frequencies are mapped correctly.

    IFFT Operation:
        The code performs an IFFT on the input buffer filled according to the 
        mapped subcarriers. The IFFT transforms the frequency domain data back 
        to the time domain, which is essential for OFDM transmission.

    Cyclic Prefix Addition:
        After performing the IFFT, a cyclic prefix is extracted from the end 
        of the IFFT output and prepended to the beginning of the OFDM symbol. 
        This is crucial for reducing inter-symbol interference in the transmission 
        environment.

    Output Sequence Assembly:
        The code concatenates each OFDM symbol (with the CP included) to form the 
        complete output sequence. The indexing is handled correctly, ensuring that 
        symbols are placed sequentially without overlap.

Efficiency

    The function operates efficiently for the task it performs, leveraging 
    NumPy's capabilities for handling array operations and FFT computations. However, 
    there might be room for optimization:
        Memory Usage: The allocation of ifft_input_buffer for each symbol might 
        be optimized if memory reuse is considered. Preallocating this buffer 
        outside the loop and resetting its values could be more efficient.
        Vectorization: Some parts of the loop might benefit from further vectorization.
        However, given the nature of the operations (especially the mapping of 
        subcarriers to the IFFT buffer), full vectorization might be complex.

Robustness

    The function uses assertions to ensure valid input parameters, which is good 
    practice for catching configuration errors early. However, assertions might 
    not be the best choice for production code due to the following reasons:
        Error Handling: Assertions raise AssertionErrors, which are not typically 
        caught in application code, leading to crashes. A more robust approach
        would involve raising specific exceptions or returning error codes/messages.
        User Feedback: For a library function, it would be more user-friendly to 
        check conditions and raise exceptions with clear messages or handle errors 
        gracefully.

Readability and Maintainability

    The code is generally well-structured and follows logical steps clearly, 
    which aids in readability. The use of meaningful variable names and the 
    separation of configuration and processing steps enhance understandability.
    Comments or docstrings describing the purpose of functions and parameters would 
    improve maintainability, especially for other developers or for future modifications.

Scalability

    The function is designed for specific subcarrier counts and sample rates. To 
    use this function in a broader range of OFDM applications, it would need to be 
    generalized to support different configurations dynamically, perhaps by 
    externalizing some of the configuration parameters.

Suggested Improvements

    Parameterize Configuration: Instead of hard-coding subcarrier indices and 
    IFFT sizes, consider passing these as parameters or defining them in a 
    configuration file or object. Improve Error Handling: Replace assertions 
    with checks that raise specific exceptions or return error statuses.
    Enhance Documentation: Add more detailed comments or docstrings explaining
    each part of the function, including its expected inputs and outputs.
        

    Returns:
        _type_: _description_
"""

import numpy as np

def ofdm_modulator(resource_grid, output_sample_rate):
    num_subcarriers, num_ofdm_symbols = resource_grid.shape

    # Error checking
    assert num_subcarriers in [913, 841], "Invalid number of subcarriers."
    assert output_sample_rate in [20e6, 40e6], "Invalid output sample rate."

    # IFFT size and CP length configuration
    if output_sample_rate == 20e6:
        ifft_size = 1024
        cp_length = 116
    else:
        ifft_size = 2048
        cp_length = 232

    ofdm_symbol_length = cp_length + ifft_size

    # Subcarrier indices setup
    if num_subcarriers == 913:
        pos_subcarrier_indices = np.arange(456, 913)
        neg_subcarrier_indices = np.arange(456)
        pos_ifft_indices = np.arange(457)
        neg_ifft_indices = np.arange(568, 1024) if ifft_size == 1024 else np.arange(1592, 2048)
    else:
        pos_subcarrier_indices = np.arange(420, 841)
        neg_subcarrier_indices = np.arange(420)
        pos_ifft_indices = np.arange(421)
        neg_ifft_indices = np.arange(604, 1024) if ifft_size == 1024 else np.arange(1628, 2048)

    # OFDM modulation process
    output_sequence = np.zeros(num_ofdm_symbols * ofdm_symbol_length, dtype=complex)
    output_sequence_index = 0

    for idx in range(num_ofdm_symbols):
        ifft_input_buffer = np.zeros(ifft_size, dtype=complex)
        ifft_input_buffer[pos_ifft_indices] = resource_grid[pos_subcarrier_indices, idx]
        ifft_input_buffer[neg_ifft_indices] = resource_grid[neg_subcarrier_indices, idx]

        ifft_output_buffer = np.fft.ifft(ifft_input_buffer)

        cyclic_prefix = ifft_output_buffer[-cp_length:]
        ofdm_symbol = np.hstack((cyclic_prefix, ifft_output_buffer))

        output_sequence[output_sequence_index:output_sequence_index + ofdm_symbol_length] = ofdm_symbol
        output_sequence_index += ofdm_symbol_length

    return output_sequence

