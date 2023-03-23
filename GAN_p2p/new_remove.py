import os
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot



# read all csv data
def read_all_csv_files(directory):
    directory = os.path.join(directory)    # Join the directory path with the os path separator
    dfs = []    # Create an empty list to store the read dataframes

    # Iterate through all the files in the directory using os.walk()
    for root, dirs, files in os.walk(directory):
        for file in files:  # Iterate through all the files and directories in the current root directory
            if file.endswith(".csv"):   # Check if the file ends with .csv
                df = pd.read_csv(os.path.join(directory, file), delimiter=',')
                dfs.append(df)  # Append the dataframe to the list of dataframes

    return dfs      # Return the list of dataframes



def apply_miss_rate_per_rf(dfs, miss_rate):
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
        no, dim = data_z.shape  # Get the number of rows and columns in the data_x array
        data_m = remove_random_columns(data_z, miss_rate)   # Call the remove_random_columns function to remove columns from data_x
        missing_data.append(data_m) # Append the missing data to the missing_data list
        full_data.append(data_z)    # Append the full data to the full_data list

    # Return the missing_data and full_data lists
    return missing_data, full_data





# Remove at random a user defined percentage of columns from the matrix
def remove_random_columns(data_z, miss_rate):
    # Transpose the input data to operate on columns instead of rows
    data_z = np.transpose(data_z)
    # Choose the dimension based on the number of columns in the transposed data
    dim_choice = int(data_z.shape[0])

    # Calculate how many columns (indexes) we need according to the missing rate
    no_missing_columns = int(miss_rate * dim_choice)
    no_columns_to_keep = abs(no_missing_columns - dim_choice)


    # Empty container for the missing indexes
    columns_to_keep_index = []
    missing_columns_index = []

    # Set the min distance between random columns
    min_distance = 2

    while len(columns_to_keep_index) != no_columns_to_keep:
        # Generate a random column index
        rand_index = int(np.random.uniform(0, dim_choice))
        if rand_index in columns_to_keep_index:
            print('already in list')
        else:
            # check if there is enough spacing
            if all(i not in columns_to_keep_index for i in range(rand_index - min_distance, rand_index + min_distance +1)):
                print('**********************there is enough space')
                columns_to_keep_index.append(rand_index)
            else:
                print('not enough space, retrying')

    # Create a matrix of ones that will be used to indicate missing data
    data_m = np.zeros_like(data_z)

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



########################################################################################################################

# Resizing images, if needed
SIZE_X = 256
SIZE_Y = 64
no_rows = SIZE_Y
no_cols = SIZE_X
path = 'C:\inpt\synthetic_data\cs2d_test'

miss_rate = 0.9

# Capture training image info as a list
tar_images = []
# Capture mask/label info as a list
src_images = []

all_csv = read_all_csv_files(path)
no_samples = len(all_csv)
missing_data, full_data= apply_miss_rate_per_rf(all_csv, miss_rate)

missing_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in missing_data])
full_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in full_data])

tar_images = np.reshape(full_data, (no_samples, no_rows, no_cols, 1))
src_images = np.reshape(missing_data, (no_samples, no_rows, no_cols, 1))

n_samples = 3
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(src_images[i], cmap='viridis')
# plot target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(tar_images[i], cmap='viridis')
pyplot.show()



# Remove at random a user defined percentage of columns from the matrix
def remove_random_columns_mod(data_z, miss_rate):
    # Transpose the input data to operate on columns instead of rows
    data_z = np.transpose(data_z)
    # Choose the dimension based on the number of columns in the transposed data
    dim_choice = int(data_z.shape[0])

    # Calculate how many columns (indexes) we need according to the missing rate
    no_missing_columns = int(miss_rate * dim_choice)




    # Randomly select a subset of columns to "remove" based on the miss_rate
    missing_columns_index = random.sample(range(dim_choice), no_missing_columns)


    if len(missing_columns_index) == no_missing_columns:
        print('they match!')
    else:
        print('keep adding!')

    # Create a matrix of ones that will be used to indicate missing data
    data_m = np.ones_like(data_z)

    # Remove a random number of rows from the bottom from each column
    data_m = remove_random_depths(data_z, data_m)

    # Set the values in data_m to 0 for the columns that were selected for removal
    for column_index in missing_columns_index:
        data_m[column_index, :] = np.zeros_like(data_m[column_index, :])

    # Multiply the original data by the missing data indicator to create the final output
    miss_list = np.multiply(data_z, data_m)
    # Transpose the output back to its original orientation
    miss_list = np.transpose(miss_list)

    return miss_list



