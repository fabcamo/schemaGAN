import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.plots import plot_input_and_target_from_dataset


def load_image_from_csv(image_path: str, height: int, width: int, channels: int) -> tf.Tensor:
    """
    Load an image from a CSV file and convert it to a TensorFlow tensor.

    Args:
        image_path (str): Path to the CSV file.
        height (int): Height of the image.
        width (int): Width of the image.
        channels (int): Number of channels in the image.

    Returns:
        image (tf.Tensor): The image tensor for a given csv file.
    """
    # Read CSV file into a DataFrame
    df = pd.read_csv(image_path)

    # Initialize an empty image with zeros
    image = np.zeros((height, width, channels), dtype=np.float32)

    # Extract coordinates and IC values
    x = df['x'].values
    z = df['z'].values
    ic = df['IC'].values

    # Populate the image with IC values at specified coordinates
    image[z, x, 0] = ic

    # Convert the numpy array to a TensorFlow tensor
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    return image


def normalize(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize the image to the range [-1, 1].

    Args:
        image (tf.Tensor): The image tensor to normalize.

    Returns:
        image (tf.Tensor): The normalized image tensor.
    """
    # Normalize pixel values to the range [-1, 1]
    image = image / 127.5 - 1

    return image


def process_image(cs_file: str, cptlike_file: str, height: int, width: int, channels: int) -> tuple:
    """
    Load and normalize images from CSV files.

    Args:
        cs_file (str): Path to the cross-section image CSV file. This is the TARGET
        cptlike_file (str): Path to the CPT-like image CSV file. this is the INPUT
        height (int): Height of the images.
        width (int): Width of the images.
        channels (int): Number of channels in the images.

    Returns:
        tuple: A tuple containing the normalized CPT-like image and normalized cross-section image tensors.
    """
    # Load and normalize the images
    cs_image = normalize(load_image_from_csv(cs_file, height, width, channels))
    cptlike_image = normalize(load_image_from_csv(cptlike_file, height, width, channels))

    return cptlike_image, cs_image


def _input_fn(inputs: tf.Tensor, target: tf.Tensor, batch_size: int) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from inputs and targets tensors. The dataset is batched according to the specified
    batch size. The function zips the input and target datasets together. It can be called for different data
    to create training, validation, and test datasets.

    Args:
        inputs (tf.Tensor): Tensor of input images.
        target (tf.Tensor): Tensor of target images.
        batch_size (int): Batch size for the dataset.

    Returns:
        dataset (tf.data.Dataset): TensorFlow dataset object. It contains batches of input-target pairs.
    """
    # Create TensorFlow Dataset objects from tensors
    input_dataset = tf.data.Dataset.from_tensor_slices(inputs)
    target_dataset = tf.data.Dataset.from_tensor_slices(target)

    # Batch the datasets
    input_dataset = input_dataset.batch(batch_size)
    target_dataset = target_dataset.batch(batch_size)

    # Zip the input and target datasets together
    dataset = tf.data.Dataset.zip((input_dataset, target_dataset))

    return dataset



def create_dataset(cs_folder: str, cptlike_folder: str, height: int, width: int, channels: int, batch_size: int,
                   test_percentage: float = None, val_percentage: float = None):
    """
    Create tensors for inputs and targets from folders containing CSV files for paired images,
    and optionally split into training, validation, and testing datasets.

    Args:
        cs_folder (str): Path to the folder containing cross-section images.
        cptlike_folder (str): Path to the folder containing CPT-like images.
        height (int): Height of the images.
        width (int): Width of the images.
        channels (int): Number of channels in the images.
        batch_size (int): Batch size for the dataset.
        test_percentage (float, optional): Percentage of data to use for testing (0 < test_percentage < 1).
        val_percentage (float, optional): Percentage of data to use for validation (0 < val_percentage < 1).

    Returns:
        tuple: A tuple containing datasets for training, validation, and testing.
    """
    # Get list of CSV files from both folders
    cs_files = sorted(str(file) for file in Path(cs_folder).glob("*.csv"))
    cptlike_files = sorted(str(file) for file in Path(cptlike_folder).glob("*.csv"))

    # Initialize empty lists to hold image tensors
    inputs, targets = [], []

    # Process each pair of files
    for cs_file, cptlike_file in zip(cs_files, cptlike_files):
        # Load and normalize the images
        cs_image = normalize(load_image_from_csv(cs_file, height, width, channels))
        cptlike_image = normalize(load_image_from_csv(cptlike_file, height, width, channels))

        # Append processed images to lists
        inputs.append(cptlike_image)  # The CPT-like image is the INPUT
        targets.append(cs_image)  # The cross-section image is the TARGET

    # Convert lists to tensors
    inputs_tensor = tf.stack(inputs)
    targets_tensor = tf.stack(targets)

    # Determine splits
    num_samples = len(inputs_tensor)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Create shuffled datasets
    inputs_tensor = tf.gather(inputs_tensor, indices)
    targets_tensor = tf.gather(targets_tensor, indices)

    # Calculate split indices
    if test_percentage is not None:
        test_size = int(num_samples * test_percentage)
    else:
        test_size = 0

    if val_percentage is not None:
        val_size = int(num_samples * val_percentage)
    else:
        val_size = 0

    # Split datasets
    if test_size > 0:
        test_inputs = inputs_tensor[:test_size]
        test_targets = targets_tensor[:test_size]
        inputs_tensor = inputs_tensor[test_size:]
        targets_tensor = targets_tensor[test_size:]
    else:
        test_inputs = test_targets = None

    if val_size > 0:
        val_inputs = inputs_tensor[:val_size]
        val_targets = targets_tensor[:val_size]
        train_inputs = inputs_tensor[val_size:]
        train_targets = targets_tensor[val_size:]
    else:
        val_inputs = val_targets = None
        train_inputs = inputs_tensor
        train_targets = targets_tensor

    # Create datasets
    train_dataset = _input_fn(train_inputs, train_targets, batch_size)
    val_dataset = _input_fn(val_inputs, val_targets, batch_size) if val_inputs is not None else None
    test_dataset = _input_fn(test_inputs, test_targets, batch_size) if test_inputs is not None else None

    return train_dataset, test_dataset, val_dataset





train_folder_cs = r"D:\GeoSchemaGen\tests\outputs\train\cs"
train_folder_cptlike = r"D:\GeoSchemaGen\tests\outputs\train\cptlike"
height = 32
width = 512
channels = 1
BATCH_SIZE = 1
TEST_PERCENTAGE = 0.2  # 20% for testing
VAL_PERCENTAGE = 0.2   # 20% for validation

# Create datasets with splits
train_dataset, test_dataset, val_dataset = create_dataset(
    train_folder_cs, train_folder_cptlike, height, width, channels,
    BATCH_SIZE, TEST_PERCENTAGE)

# Print dataset information
for batch in train_dataset.take(1):
    input_batch, target_batch = batch
    print(f'Train Input batch shape: {input_batch.shape}')
    print(f'Train Target batch shape: {target_batch.shape}')

if val_dataset:
    for batch in val_dataset.take(1):
        input_batch, target_batch = batch
        print(f'Validation Input batch shape: {input_batch.shape}')
        print(f'Validation Target batch shape: {target_batch.shape}')

if test_dataset:
    for batch in test_dataset.take(1):
        input_batch, target_batch = batch
        print(f'Test Input batch shape: {input_batch.shape}')
        print(f'Test Target batch shape: {target_batch.shape}')


# Count elements in each dataset
num_train_batches = sum(1 for _ in train_dataset)
num_val_batches = sum(1 for _ in val_dataset) if val_dataset else 0
num_test_batches = sum(1 for _ in test_dataset) if test_dataset else 0

# Print the number of elements
print(f'Training dataset size: {num_train_batches * BATCH_SIZE}')
if val_dataset:
    print(f'Validation dataset size: {num_val_batches * BATCH_SIZE}')
if test_dataset:
    print(f'Test dataset size: {num_test_batches * BATCH_SIZE}')




# Example usage
num_samples = 5  # Number of pairs to plot
plot_input_and_target_from_dataset(train_dataset, num_samples)