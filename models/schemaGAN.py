import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

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
    # Define column types for x, z, and IC
    column_defaults = [tf.int32, tf.int32, tf.float32]

    # Create a CsvDataset, skipping the header and handling the unnamed index
    dataset = tf.data.experimental.CsvDataset(image_path,
                                              record_defaults=column_defaults,
                                              header=True,
                                              select_cols=[1, 2, 3])

    # Initialize an empty tensor for the image
    image = tf.zeros([height, width, channels], dtype=tf.float32)

    # Loop through the dataset and update the tensor with the IC values
    for x, z, ic_value in dataset:
        image = tf.tensor_scatter_nd_update(image, [[z, x, 0]], [ic_value])

    return image

def load_paired_images(cs_path: str, cptlike_path: str, height: int, width: int, channels: int):
    """
    Load paired cross-section and CPT-like images from CSV files containing the IC values.

    Args:
        cs_path (str): The path to the cross-section image file.
        cptlike_path (str): The path to the CPT-like image file.
        height (int): The height of the image.
        width (int): The width of the image.
        channels (int): The number of channels in the image.

    Returns:
        cs_image (tf.Tensor): The cross-section image tensor.
        cptlike_image (tf.Tensor): The CPT-like image tensor.
    """
    cptlike_image = load_image_from_csv(cptlike_path, height, width, channels)
    cs_image = load_image_from_csv(cs_path, height, width, channels)

    return cptlike_image, cs_image

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

def load_image_train(cs_file: tf.Tensor, cptlike_file: tf.Tensor, height: int, width: int, channels: int):
    """
    Load and normalize training images from CSV files.

    Args:
        cs_file (tf.Tensor): Path to the cross-section image CSV file.
        cptlike_file (tf.Tensor): Path to the CPT-like image CSV file.

    Returns:
        cptlike_image (tf.Tensor): The normalized CPT-like image tensor.
        cs_image (tf.Tensor): The normalized cross-section image tensor.
    """
    # Convert the paths to strings
    cs_file = cs_file.numpy().decode('utf-8')
    cptlike_file = cptlike_file.numpy().decode('utf-8')

    # Load the images from the CSV files
    cptlike_image, cs_image = load_paired_images(cs_file, cptlike_file, height, width, channels)

    # Normalize the images
    cptlike_image = normalize(cptlike_image)
    cs_image = normalize(cs_image)

    return cptlike_image, cs_image

def create_dataset(cs_folder: str, cptlike_folder: str, height: int, width: int, channels: int, buffer_size: int, batch_size: int):
    """
    Create a TensorFlow dataset from a folder of paired cross-section and CPT-like images. The images are loaded from
    CSV files containing the IC values. The dataset is shuffled and batched. The images are normalized to the range
    [-1, 1]. The dataset is returned with the specified buffer size and batch size.

    Args:
        cs_folder (str): The path to the folder containing the cross-section images.
        cptlike_folder (str): The path to the folder containing the CPT-like images.
        height (int): The height of the images.
        width (int): The width of the images.
        channels (int): The number of channels in the images.
        buffer_size (int): The buffer size for shuffling the dataset.
        batch_size (int): The batch size for the dataset.

    Returns:
        dataset (tf.data.Dataset): The TensorFlow dataset containing the paired cross-section and CPT-like images.
    """
    # Get a list of all the CSV files in both folders
    cs_files = sorted(Path(cs_folder).glob("*.csv"))
    cptlike_files = sorted(Path(cptlike_folder).glob("*.csv"))

    # Create a dataset from the list of files
    # With from_tensor_slices, the dataset will return pairs of paths to the CSV files
    dataset = tf.data.Dataset.from_tensor_slices((list(map(str, cs_files)), list(map(str, cptlike_files))))
    # With map, we can apply a function to each pair of paths, in this case, load_image_train
    dataset = dataset.map(lambda x, y: tf.py_function(
                                        func=load_image_train,
                                        inp=[x, y, height, width, channels],
                                        Tout=[tf.float32, tf.float32]),
                                        num_parallel_calls=tf.data.AUTOTUNE)
    # Ensure the shape of the images
    dataset = dataset.map(lambda cptlike, cs: (tf.ensure_shape(cptlike, [height, width, channels]), tf.ensure_shape(cs, [height, width, channels])))

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset

def observe_dataset_samples(dataset, num_samples: int):
    for input_image, real_image in dataset.take(num_samples):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(input_image[0, :, :, 0].numpy(), cmap='viridis')
        plt.title("Input Image")
        plt.subplot(1, 2, 2)
        plt.imshow(real_image[0, :, :, 0].numpy(), cmap='viridis')
        plt.title("Real Image")
        plt.show()

# Parameters
train_folder_cs = r"D:\GeoSchemaGen\tests\outputs\train\cs"
train_folder_cptlike = r"D:\GeoSchemaGen\tests\outputs\train\cptlike"
height = 32
width = 512
channels = 1
BUFFER_SIZE = 80
BATCH_SIZE = 1

# Create dataset
train_dataset = create_dataset(train_folder_cs, train_folder_cptlike, height, width, channels, BUFFER_SIZE, BATCH_SIZE)

# Plot dataset samples
observe_dataset_samples(dataset=train_dataset, num_samples=3)

print(train_dataset)
