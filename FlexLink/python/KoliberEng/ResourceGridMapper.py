"""
first create an empty resource grid with the specified dimensions.
Then check if the number of IQ values in the input list matches the size 
of the grid (i.e., num_subcarriers * num_symbols).
We then iterate through the IQ values and map each IQ symbol to a 
resource element in the grid, filling it column by column.
Finally, return the resulting resource grid.

calculate the total number of available resource elements and ensure 
that the number of IQ values passed matches this count. 

create an empty resource grid with the specified dimensions and use nested 
loops to fill it. The first column is filled with pilot signals p1, p0, and 
IQ values. The remaining columns are filled with IQ symbols from the IQ values list. We 
adjust the example list of IQ values to match the available resource elements.

The repeating pattern defined by the pattern list, which specifies
the order of elements ("p1," "p0," and "IQ"). We use the pattern_idx variable 
to keep track of the current pattern element and update it for the next iteration.
The first column is filled with pilots "p1" and "p0," and the remaining 
columns are filled with IQ symbols from the IQ values list based on the defined pattern.
"""



import numpy as np

def map_iq_symbols_to_resource_grid(iq_values, num_subcarriers, num_symbols, pattern):
    # Calculate the number of available resource elements
    total_resource_elements = num_subcarriers * num_symbols
    
    # Check if the number of IQ values matches the available resource elements
    if len(iq_values) != total_resource_elements - int(num_subcarriers/3):
        raise ValueError("Number of IQ values does not match the available resource elements.")
    
    # Create an empty resource grid (2D NumPy array)
    # np.zeros((N-rows, N-columns), dtype)
    resource_grid = np.zeros((num_subcarriers, num_symbols), dtype=complex)
    
    # Initialize the pattern index
    pattern_idx = 0
    
    # Iterate through the IQ values and map each to a resource element
    for symbol_idx in range(num_symbols):
        for subcarrier_idx in range(num_subcarriers):
            # Get the current pattern element (p1, p0, or IQ)
            current_element = pattern[pattern_idx]
            # Update the pattern index for the next iteration
            pattern_idx = (pattern_idx + 1) % len(pattern)
            
            # If it's p1 or p0, fill the first column accordingly
            if symbol_idx == 0:
                if current_element == "p1":
                    resource_grid[subcarrier_idx, symbol_idx] = complex( 1, 0)  # p1
                elif current_element == "p0":
                    resource_grid[subcarrier_idx, symbol_idx] = complex(-1, 0)  # p0
                else: 
                    # Get the IQ value from the list
                    iq_symbol = iq_values.pop(0)
                    resource_grid[subcarrier_idx, symbol_idx] = iq_symbol
            else: # rest of resource grid columns
                # Get the IQ value from the list
                iq_symbol = iq_values.pop(0)
                
                # Assign the IQ symbol to the resource grid
                resource_grid[subcarrier_idx, symbol_idx] = iq_symbol
            
    return resource_grid

# Example list of IQ values (adjusted to fit the available resource elements)
num_subcarriers = 16
num_symbols = 5
available_resource_elements = num_subcarriers * num_symbols
iq_values = [complex(0.5, 0)] * (available_resource_elements - int(num_subcarriers/3))  # Adjusted IQ symbols

# Define the repeating pattern (p1, p0, IQ)
pattern = ["p1", "p0", "IQ"]

# Map the IQ symbols to the resource grid based on the pattern
resulting_grid = map_iq_symbols_to_resource_grid(iq_values, num_subcarriers, num_symbols, pattern)

# Print the resulting resource grid
print(resulting_grid)
