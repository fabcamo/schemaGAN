import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from models.generator import Generator_modular
from models.discriminator import Discriminator_modular


def load_images_from_csv(cs_path: str, cptlike_path: str, height: int = 32, width: int = 512, channels: int = 1):
    """
    Load images from CSV files containing the IC values for cross-section and CPT-like images.

    Args:
        cs_path (str): The path to the cross-section image file.
        cptlike_path (str): The path to the CPT-like image file.
        height (int, optional): The height of the image. Default is 32.
        width (int, optional): The width of the image. Default is 512.
        channels (int, optional): The number of channels in the image. Default is 1.

    Returns:
        tuple: A tuple containing two image tensors (cs_image, cptlike_image).
    """
    def load_image_from_csv(csv_file_path):
        """
        Load an image from a CSV file containing the IC values.

        Args:
            csv_file_path (str): The path to the CSV file.

        Returns:
            tf.Tensor: The image tensor containing the IC values.
        """
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)

        # Initialize an empty tensor for the image
        image = tf.zeros([height, width, channels], dtype=tf.float32)

        # Populate the tensor with the IC values
        for _, row in df.iterrows():
            x, z, ic_value = int(row['x']), int(row['z']), float(row['IC'])
            image = tf.tensor_scatter_nd_update(image, [[z, x, 0]], [ic_value])

        return image

    # Load cross-section image
    cs_image = load_image_from_csv(cs_path)

    # Load CPT-like image
    cptlike_image = load_image_from_csv(cptlike_path)

    return cs_image, cptlike_image



def normalize(image: tf.Tensor):
    """
    Normalize the image to the range [-1, 1]. This is the range expected by the generator model.

    Args:
        image (tf.Tensor): The image tensor to normalize.

    Returns:
        tf.Tensor: The normalized image tensor.

    """
    normalized_image = (image / 127.5) - 1

    return normalized_image


def load_image_train(cs_file, cptlike_file):
    """
    Load and normalize training images from CSV files.

    Args:
        cs_file (str): Path to the cross-section image CSV file.
        cptlike_file (str): Path to the CPT-like image CSV file.

    Returns:
        tuple: Normalized cross-section and CPT-like image tensors.
    """
    cs_image, cptlike_image = load_images_from_csv(cs_file, cptlike_file)
    cs_image = normalize(cs_image)
    cptlike_image = normalize(cptlike_image)
    return cs_image, cptlike_image




PATH =

# List files in the train directory
input_files = sorted(PATH.glob('train/cs_*.jpg'))
real_files = sorted(PATH.glob('train/cptlike_*.jpg'))

# Create a dataset of file pairs
file_pairs = list(zip(input_files, real_files))

# Create a TensorFlow Dataset from the file pairs
train_dataset = tf.data.Dataset.from_tensor_slices((file_pairs))
train_dataset = train_dataset.map(lambda input_file, real_file: tf.py_function(
    func=lambda input_file, real_file: load_image_train(input_file.numpy().decode('utf-8'), real_file.numpy().decode('utf-8')),
    inp=[input_file, real_file],
    Tout=[tf.float32, tf.float32]),
    num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# Example usage:
show_samples_from_dataset(train_dataset, num_samples=3)



BUFFER_SIZE = 400  # Example buffer size, adjust as needed
BATCH_SIZE = 1     # Example batch size, adjust as needed

# Example usage:
cs_path = "D:/GeoSchemaGen/tests/outputs/train/cs_10.csv"
cptlike_path = "D:/GeoSchemaGen/tests/outputs/train/cptlike_10.csv"

dataset = create_dataset("D:/GeoSchemaGen/tests/outputs/train")

print(dataset)



def test_plot_image(image):
    # Squeeze the last dimension for visualization purposes if it has only one channel
    image_np = tf.squeeze(image).numpy()

    # Plot the image
    plt.imshow(image_np, cmap='gray')
    plt.title("Loaded Image")
    plt.colorbar()
    plt.show()







##### FROM HERE ON IT'S ELENI'S CODE ############################



def _input_fn(inputs, target):

    input_dataset = tf.convert_to_tensor(tf.constant(inputs))
    train_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset)
    train_input_dataset = train_input_dataset.batch(BATCH_SIZE)
    target_dataset = tf.convert_to_tensor(tf.constant(target))
    train_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset)
    train_target_dataset = train_target_dataset.batch(BATCH_SIZE)
    train_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))
    return train_dataset

def load_from_pickle(file):
    model = pickle.load(open(file, "rb"))
    feature_names = model['feature_names']
    all_data = model['all_data']
    return feature_names, np.array(all_data).astype(np.float32)

def set_up_data_as_input(directory_inputs):
    pickle_names = [o for o in os.listdir(directory_inputs) if os.path.isfile(os.path.join(directory_inputs,o)) and o.endswith(".p")]
    global_data = []
    for pickle_name in pickle_names:
        feature_names_loc, all_data = load_from_pickle(os.path.join(directory_inputs, pickle_name))
        if len(feature_names_loc) == 6:
            feature_names = feature_names_loc
        global_data.append(all_data)
    # remove empty lists
    global_data = [data for data in global_data if len(data) > 0]
    dataset_all = np.concatenate(global_data, axis=0)
    # find indexes for input
    inputs_lookup = ['FRICTION_ANGLE', 'geometry', 'water_level', 'YOUNGS_MODULUS']
    inputs_indexes = [feature_names.index(input) for input in inputs_lookup]
    # find indexes for output
    outputs_lookup = ['total_displacement_stage_3']
    outputs_indexes = [feature_names.index(output) for output in outputs_lookup]
    inputs_dataset = dataset_all[:, inputs_indexes, :, :]
    outputs_dataset = dataset_all[:, outputs_indexes, :, :]
    # reshape the data from (number_of_inputs, number_of_features, 256, 256) to (number_of_inputs, 256, 256, number_of_features)
    inputs_dataset = np.moveaxis(inputs_dataset, 1, -1)
    outputs_dataset = np.moveaxis(outputs_dataset, 1, -1)

    # split the data into train and test
    percentage_train = 0.8
    number_of_inputs = inputs_dataset.shape[1]
    train_input_dataset = inputs_dataset[int(percentage_train * number_of_inputs):, :, :, :]
    train_output_dataset = outputs_dataset[int(percentage_train * number_of_inputs):]
    test_input_dataset = inputs_dataset[int(percentage_train * number_of_inputs):, :, :, :]
    test_output_dataset = outputs_dataset[int(percentage_train * number_of_inputs):, :, :]

    train_dataset = _input_fn(train_input_dataset, train_output_dataset)
    test_dataset = _input_fn(test_input_dataset, test_output_dataset)
    return train_dataset, test_dataset



def set_up_and_train_2d_model():
    directory_inputs = "D:/schemaGAN/data"
    train_dataset, test_dataset = set_up_data_as_input(directory_inputs)

    fit(train_dataset, test_dataset, None,  steps=50000)

    # TODO plot against all the training inputs
    all_images(generator, test_dataset)
    # TODO plot diff
    dataset = [value for counter, value in enumerate(test_dataset)]
    predictions = [generator(data[0], training=True) for data in dataset]
    data_diff = [data[1].numpy() - predictions[counter].numpy() for counter, data in enumerate(dataset)]

    mean_axis_error = np.mean(np.array(np.abs(data_diff)), axis=0)
    plt.clf()
    plt.imshow(mean_axis_error)
    plt.colorbar
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(mean_axis_error.T)
    ax.set_xlabel("Mean error per pixel at the end of training")
    fig.colorbar(im, orientation="horizontal")
    plt.show()

    # TODO calculate all the diffs
    mean_error = np.mean(np.abs(np.array(data_diff)).flatten())
    std_error = np.std(np.array(data_diff).flatten())
    print(f"stats mean abserror {mean_error} std : {std_error}")




"""
if __name__ == "__main__":
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment


    # define optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    log_dir = "logs_geometry_re/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit_train_geometry_re/keep" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    loss_object = tf.keras.losses.MeanSquaredError()

    # Create the generator and discriminator models
    generator = Generator_modular()
    discriminator = Discriminator_modular()

    checkpoint_dir = "D:/schemaGAN/tests"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    # create output directory if it does not exist
    if not os.path.exists("output_geometry_re"):
        os.makedirs("output_geometry_re")

    set_up_and_train_2d_model()

"""