import tensorflow as tf

import time
import datetime
import numpy as np
import plotly.express as px
from PIL import Image

from models.generator import Generator_modular
from models.discriminator import Discriminator_modular
from models.training import train_step, fit

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from IPython import display
import pickle
import os

def all_images(model, input_dataset):
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Times'], 'size': 14})
    rc('text', usetex=True)
    dataset = [value for counter, value in enumerate(input_dataset)]
    predictions = [model(data[0], training=True) for data in dataset]
    data_diff = [data[1].numpy()[0,:,:] - predictions[counter].numpy()[0,:,:,0] for counter, data in enumerate(dataset)]
    for counter, image_output in enumerate(predictions):
        fig, ax = plt.subplots(1, 4)
        fig.set_size_inches(18.5, 10.5)
        error = np.mean(np.array(np.abs(data_diff[counter])))
        fig.suptitle(f"Absolute error: {error}", fontsize=15)
        display_list = [dataset[counter][0][0].numpy().T * AMAX,
                        dataset[counter][1][0].numpy().T * AMAX,
                        predictions[counter].numpy().T[0,:,:] * AMAX,
                        data_diff[counter].T * AMAX]
        title = ["Input Image", "Ground Truth", "Predicted Image", "Absolute difference"]
        for i in range(4):
            ax[i].set_title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            im = ax[i].imshow(display_list[i], vmin=0, vmax=AMAX)
            ax[i].axis("off")
        fig.colorbar(im, ax=ax.ravel().tolist(), orientation="horizontal")
        fig.savefig(f"slice_{counter}_prediction.png")
        plt.close(fig)


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
    directory_inputs = "D:/sheetpile/ai_model/inputs_geometry_re/"
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


BATCH_SIZE = 1
import os
AMAX = 4.5
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
OUTPUT_CHANNELS = 1
LAMBDA = 100
IMAGE_SIZE = 256
IMAGE_IDX = 255




if __name__ == "__main__":
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment

    # Define optimizers for the generator and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

    # Define the log directory
    log_dir = "logs_geometry_re/"

    # Create a summary writer
    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit_train_geometry_re/keep" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # Define the loss object to be used for calculating the loss
    loss_object = tf.keras.losses.MeanSquaredError()

    # Define the generator models
    generator = Generator_modular(input_size=(256, 256),
                                  no_inputs=4,
                                  OUTPUT_CHANNELS=1,
                                  base_filters=64,
                                  dropout_layers=3)

    # Define the discriminator model
    discriminator = Discriminator_modular(input_size=(256, 256),
                                          no_inputs=4,
                                          base_filters=64)

    # Define the checkpoint directory
    checkpoint_dir = "D:\sheetpile\\ai_model\\training_checkpoints_2d_geometry_refined"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )


    # Create output directory if it does not exist
    if not os.path.exists("output_geometry_re"):
        os.makedirs("output_geometry_re")

    # Setup and train the model
    set_up_and_train_2d_model()
