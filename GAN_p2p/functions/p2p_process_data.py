import os
import numpy as np
import pandas as pd



# Read all csv data to train on
def read_all_csv_files(directory):
    directory = os.path.join(directory)    # Join the directory path with the os path separator
    csv_data = []    # Create an empty list to store the read dataframes

    # Iterate through all the files in the directory using os.walk()
    for root, dirs, files in os.walk(directory):
        for file in files:  # Iterate through all the files and directories in the current root directory
            if file.endswith(".csv"):   # Check if the file ends with .csv
                df = pd.read_csv(os.path.join(directory, file), delimiter=',')
                csv_data.append(df)  # Append the dataframe to the list of dataframes

    return csv_data      # Return the list of dataframes



# Create the conditional input CPT-like data
def apply_miss_rate_per_rf(dfs, miss_rate, min_distance):
    missing_data, full_data = [], []     # Create two empty lists to store missing and full data
    value_name = 'IC'   # Set value_name to 'IC'

    # Iterate through each random field in the list
    for counter, rf in enumerate(dfs):
        data_z = []     # Create an empty list to store data for each value of x
        grouped = rf.groupby("z")   # Group the rows of the random field by the value of x

        # Iterate through each group
        for name, group in grouped:
            data_z.append(list(group[value_name]))  # Append the 'IC' column of the group to the data_x list

        data_z = np.array(data_z, dtype=float)  # Convert the data_x list to a numpy array of type float
        data_m = remove_random_columns(data_z, miss_rate, min_distance)   # Call the remove_random_columns function to remove columns from data_x
        missing_data.append(data_m) # Append the missing data to the missing_data list
        full_data.append(data_z)    # Append the full data to the full_data list

    # Return the missing_data and full_data lists
    return missing_data, full_data




# Remove at random a user defined percentage of columns from the matrix
def remove_random_columns(data_z, miss_rate, min_distance):
    # Transpose the input data to operate on columns instead of rows
    data_z = np.transpose(data_z)
    # Create a matrix of ones that will be used to indicate missing data
    data_m = np.zeros_like(data_z)

    # Returns which columns to keep from miss_rate and min_distance
    columns_to_keep_index = check_min_spacing(data_z, miss_rate, min_distance)

    # Set the values in data_m to 0 for the columns that were selected for removal
    for column_index in columns_to_keep_index:
        data_m[column_index, :] = np.ones_like(data_m[column_index, :])

    # Remove a random number of rows from the bottom from each column
    data_m = remove_random_depths(data_z, data_m)

    # Multiply the original data by the missing data indicator to create the final output
    miss_list = np.multiply(data_z, data_m)
    # Transpose the output back to its original orientation
    miss_list = np.transpose(miss_list)

    return miss_list



# Select the columns to keep for each cross-section
def check_min_spacing(data_z, miss_rate, min_distance):
    # Choose the dimension based on the number of columns in the transposed data
    all_columns = int(data_z.shape[0])  # [256]
    # Calculate how many columns (indexes) we need according to the missing rate
    no_missing_columns = int(miss_rate * all_columns)  # number of missing columns
    no_columns_to_keep = abs(no_missing_columns - all_columns)  # number of columns to keep [like CPTs]

    columns_to_keep_index = []  # Empty container for the missing indexes

    # Loop until the columns_to_keep_index list is full according to the selected percentage
    while len(columns_to_keep_index) != no_columns_to_keep:
        # Generate a random column index from an uniform distribution
        rand_index = int(np.random.uniform(0, all_columns))
        # Define the range of indexes to check for duplicates according to the min_distance defined
        range_to_check = range(rand_index - min_distance, rand_index + min_distance + 1)

        if rand_index in columns_to_keep_index:  # Check if the rand_index is already in columns_to_keep_index
            pass  # if it is> do nothing, restart the while-loop
        else:
            # Check if none of the indexes in the range are already in columns_to_keep_index list
            if all(index not in columns_to_keep_index for index in range_to_check):
                columns_to_keep_index.append(rand_index)  # if true, append the rand_index to the list
            else:
                print('No space to accommodate random index, RETRYING')

    return columns_to_keep_index



# Remove a random amount of data from the bottom of each column in the matrix
def remove_random_depths(data_z, data_m):
    data_length = data_z.shape[0]  # grab the length of the cross-section [256 columns]
    data_depth = data_z.shape[1]   # grab the depth of the cross-section [64 rows]
    for j in range(data_length):   # iterate over columns
        # generate a random number with bias towards lower numbers
        # this will be the number of rows to transform to zero
        n_rows = int(np.random.triangular(0,0, data_depth/2))
        if n_rows > 0:
            # for every j column, select the last n_rows
            # replace those n_rows with zeros
            data_m[j, -n_rows:] = np.zeros(n_rows)

    return data_m



# Normalize data
def preprocess_data(data):
    # load compressed arrays
    # unpack arrays
    X1, X2 = data[0], data[1]
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]
