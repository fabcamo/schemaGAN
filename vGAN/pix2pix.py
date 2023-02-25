import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import numpy as np



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




def apply_miss_rate_per_rf(dfs, miss_rate=0.8):
    missing_data, full_data = [], []     # Create two empty lists to store missing and full data
    value_name = 'IC'   # Set value_name to 'IC'

    # Iterate through each random forest in the list
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


def load_and_normalize_RFs_in_folder(directory):
    dfs = read_all_csv_files(directory)
    train_input, train_output = apply_miss_rate_per_rf(dfs)
    # Careful, here is 64 rows and 256 columns
    train_input = np.array([np.reshape(i, (64, 256)).astype(np.float32) for i in train_input])
    train_output = np.array([np.reshape(i, (64, 256)).astype(np.float32) for i in train_output])
    maximum_value = max_IC_value
    train_output = np.array(train_output) / maximum_value
    train_input = np.array(train_input) / maximum_value
    return np.array([train_input, train_output]).astype(np.float32)
    # train_input > the RFs with 80% missing (the input to generate later with AI)
    # train_output > the complete RFs

def remove_random_columns(data_z, miss_rate):
    dim_choice = int(data_z.shape[0])
    missing_columns_index = random.sample(range(dim_choice), int(miss_rate*dim_choice))
    data_m = np.ones_like(data_z)
    for column_index in missing_columns_index:
        data_m[column_index, :] = np.zeros_like(data_m[column_index, :])
    miss_list = np.multiply(data_z, data_m)
    return miss_list

#####################################################################################################

path = '/inpt/synthetic_data/cs2d/'
directory = '/inpt/synthetic_data/cs2d/'

#img = mpimg.imread(f'{path}cs_50.png')
#plt.imshow(img)
#plt.show()

data = read_all_csv_files(directory)
merged_data =  pd.concat(data)
max_IC_value = merged_data["IC"].max()
print("The maximum value of the 'IC' column in the list of dataframes is:", max_IC_value)



missing_data, full_data = apply_miss_rate_per_rf(data)

data_set = load_and_normalize_RFs_in_folder(directory)

print('train input normalized')
print(data_set[1][0].shape)
test = data_set[1][0]
print(test)
# Swap the x-axis and y-axis using transpose()

plt.imshow(test)
plt.colorbar()
plt.show()

BATCH_SIZE = 1
#AMAX = 4.5
OUTPUT_CHANNELS = 1
LAMBDA = 100
# define optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
