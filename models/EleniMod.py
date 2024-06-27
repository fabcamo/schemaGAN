import os
import pickle
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.generator import Generator_modular
from models.discriminator import Discriminator_modular
from models.training import train_step, fit


def _input_fn(inputs, target):
    """
    Create a dataset from the input and target data. The input data and target data are converted to tensors and
    sliced. The input and target data are batched and zipped together to create the training dataset.

    Args:
        inputs: The input data.
        target: The target data.

    Returns:
        train_dataset: The training dataset.

    """

    # Convert the input and target data to a tensor
    input_dataset = tf.convert_to_tensor(tf.constant(inputs))
    target_dataset = tf.convert_to_tensor(tf.constant(target))

    # Slice the input data and target data
    train_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset)
    train_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset)

    # Batch the input and target data
    train_input_dataset = train_input_dataset.batch(BATCH_SIZE)
    train_target_dataset = train_target_dataset.batch(BATCH_SIZE)

    # Zip the input and target data together
    train_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))

    return train_dataset


def load_from_pickle(file: str):
    """
    Load the data from a pickle file. The function loads the feature names and the data from the pickle file and returns
    the feature names and the data. The data is converted to a numpy array and cast to a float32.

    Args:
        file (str): The path to the pickle file.

    Returns:
        freature_names (list): The list of feature names.
        all_data (np.array): The data as a numpy array.
    """
    # Load the data from the pickle file
    model = pickle.load(open(file, "rb"))
    # Get the feature names and the data from the model
    feature_names = model['feature_names']
    all_data = model['all_data']

    # Convert the data to a numpy array and cast to a float32
    all_data = np.array(all_data).astype(np.float32)

    return feature_names, all_data



def set_up_data_as_input(inputs_path: str):
    """
    Set up the data as input for the model. The data is loaded from the input path and split into a training and test
    dataset. The input data is reshaped from (number_of_inputs, number_of_features, 256, 256) to
    (number_of_inputs, 256, 256, number_of_features). The data is split into training and test datasets based on the
    percentage_train variable. The input and output data is extracted from the dataset and the input data is split into
    training and test datasets. The input and output data is then passed to the _input_fn function to create the
    training and test datasets. The function returns the training and test datasets.

    Args:
        inputs_path (str): The path to the input data.

    Returns:

    """

    # Get the list of files in the input path that end with .p (pickle files)
    pickle_names = [filename for filename in os.listdir(inputs_path
                                                        ) if os.path.isfile(os.path.join(inputs_path, filename)
                                                                            ) and filename.endswith(".p")]

    # Create empty container
    global_data = []

    # loop over all the pickles and load the data
    for pickle_name in pickle_names:
        # load the name of the features and the data from the pickle
        feature_names_loc, all_data = load_from_pickle(os.path.join(inputs_path, pickle_name))
        # if the number of features is 6, then we have the correct data
        if len(feature_names_loc) == 6:
            # set the feature names to the feature names from the pickle file
            feature_names = feature_names_loc
        # append the data to the global data
        global_data.append(all_data)


    # remove empty lists
    global_data = [data for data in global_data if len(data) > 0]

    # concatenate the data to one big dataset
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
    # Define the directory containing the input images
    inputs_path = "D:/sheetpile/ai_model/inputs_geometry_re/"

    # Set up the training and test datasets
    train_dataset, test_dataset = set_up_data_as_input(inputs_path)

    #############################################################################################################
    # TRAIN THE MODEL BY CALLING THE FIT FUNCTION
    fit(train_ds=train_dataset, test_ds=test_dataset, validation_ds=None,  steps=50000)
    #############################################################################################################

    # Plots the progress of the model against all the training inputs
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





##### USER DEFINED VARIABLES ####################################################################################
BATCH_SIZE = 1
OUTPUT_CHANNELS = 1
LAMBDA = 100
IMAGE_SIZE = 256
IMAGE_IDX = 255

if __name__ == "__main__":

    # Define optimizers for the generator and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

    summary_writer = tf.summary.create_file_writer(
        "logs_geometry_re/fit_train_geometry_re/keep" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # Define the generator
    generator = Generator_modular(input_size=(256, 256),
                                  no_inputs=4,
                                  OUTPUT_CHANNELS=1,
                                  base_filters=64,
                                  dropout_layers=3)

    # Define the discriminator
    discriminator = Discriminator_modular(input_size=(256, 256),
                                          no_inputs=4,
                                          base_filters=64)

    # Define the checkpoint directory
    checkpoint_dir = "D:/schemaGAN/tests/checkpoints"
    # Define the checkpoint prefix to save the model
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # Define the checkpoint
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    # Create the output directory if it does not exist
    if not os.path.exists("output_geometry_re"):
        os.makedirs("output_geometry_re")

    # Train the model
    set_up_and_train_2d_model()