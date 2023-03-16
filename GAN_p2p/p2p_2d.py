import numpy as np
import pandas as pd
import tensorflow as tf

import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

directory = '/inpt/synthetic_data/cs2d/train'



# Read the data
dfs = []    # Create an empty list to store the read dataframes
# Iterate through all the files in the directory using os.walk()
for root, dirs, files in os.walk(directory):
    for file in files:  # Iterate through all the files and directories in the current root directory
        if file.endswith(".csv"):   # Check if the file ends with .csv
            df = pd.read_csv(os.path.join(directory, file), delimiter=',')
            dfs.append(df)  # Append the dataframe to the list of dataframes

print('dfs')
print(dfs)

# Extract only the IC data from each dataframe
train_data = []
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
    train_data.append(data_z)    # Append the full data to the full_data list

# Reshape the train data
train_data = np.array([np.reshape(i, (64, 256)).astype(np.float32) for i in train_data])
print('train data')
print(train_data)

# normalize the data
max_IC_value = train_data.max()
print('the max IC value is:', max_IC_value)
maximum_value = max_IC_value
train_data = np.array(train_data) / maximum_value

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# THE GENERATOR
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model