import pandas as pd
from p2p_process_data import read_all_csv_files


# Path the the data
path = 'C:\\inpt\\synthetic_data\\512x32'

# Load the data
all_csv = read_all_csv_files(path)


def find_max_min_IC(all_csv):
    max_value = float('-inf')              # Initialize the maximum value to negative infinity
    min_value = float('inf')               # Initialize the minimum value to positive infinity

    # Iterate through all the DataFrames in all_csv
    for df in all_csv:
        file_max_value = df['IC'].max()  # Find the maximum value in the 'IC' column of the current DataFrame
        file_min_value = df['IC'].min()  # Find the minimum value in the 'IC' column of the current DataFrame
        if file_max_value > max_value:   # Update the maximum value if necessary
            max_value = file_max_value
        if file_min_value < min_value:   # Update the minimum value if necessary
            min_value = file_min_value

    return (max_value, min_value)


max_value, min_value = find_max_min_IC(all_csv)
print("Max value:", max_value)
print("Min value:", min_value)


