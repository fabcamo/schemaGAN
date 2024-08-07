# in this code I am trying to recreate the creation of a database from pickle files
# that Eleni wrote for her GAN, to use the basix pix2pix code. I am not using
# pickle files but csv files.

import os
import time
import tensorflow as tf
import pandas as pd
import numpy as np

# First is the code to read the csv file
def load_image_from_csv(image_path: str, height: int, width: int, channels: int):
    """
    Load an image from a CSV file containing the IC values. The CSV file is expected to have the following columns:
    - unnamed: The index of the row.
    - x: The x-coordinate of the IC value.
    - z: The z-coordinate of the IC value.
    - IC: The IC value.

    Args:
        image_path (str): The path to the CSV file as a string.
        height (int): The height of the image.
        width (int): The width of the image.
        channels (int): The number of channels in the image.

    Returns:
        image (tf.Tensor): The image tensor.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(image_path)

    # Initialize an empty numpy array for the image
    image = np.zeros((height, width, channels), dtype=np.float32)

    # Extract x, z, and IC values
    x = df['x'].values
    z = df['z'].values
    ic = df['IC'].values

    # Update the numpy array with the IC values
    image[z, x, 0] = ic

    # Convert the numpy array to a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    return image_tensor


def normalize(input_image: tf.Tensor):
    """
    Normalize the input image to the range [-1, 1].

    Args:
        input_image (tf.Tensor): The input image tensor.

    Returns:
       input_image (tf.Tensor): The normalized input image tensor.
    """
    input_image = tf.cast(input_image, tf.float32)
    input_image = (input_image / 127.5) - 1
    return input_image


def set_up_database(data_folder_path: str):
    """
    Placeholder.

    Args:
        data_folder_path:

    Returns:

    """
    # Find all the CSV files in the data folder
    csv_file_names = [file for file in os.listdir(data_folder_path) if file.endswith('.csv')]



# Parameters
train_folder_cs = r"D:\GeoSchemaGen\tests\outputs\train\cs"
train_folder_cptlike = r"D:\GeoSchemaGen\tests\outputs\train\cptlike"
height = 32
width = 512
channels = 1
BUFFER_SIZE = 80
BATCH_SIZE = 1

image_no_1 = r'D:\GeoSchemaGen\tests\outputs\train\cs\cs_1.csv'


# create a timer
start = time.time()
image_new = load_image_from_csv(image_no_1, height, width, channels)
end = time.time()
print("New method took: ", end - start)
print(image_new)

# Plot dataset samples
#observe_dataset_samples(dataset=train_dataset, num_samples=3)

#print(train_dataset)