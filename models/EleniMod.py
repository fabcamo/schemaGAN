import os
import pickle
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from IPython import display
from models.generator import Generator_modular
from models.discriminator import Discriminator_modular
#from models.training import train_step, fit
from utils.preprocessing import create_dataset



@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )
    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=step)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step)
        tf.summary.scalar("disc_loss", disc_loss, step=step)



def generate_images(model, test_input, tar, savefig=True, step=None):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0, :, :, 0].numpy(), tar[0, :, :].numpy(), prediction[0, :, :, 0].numpy()]
    title = ["Input Image", "Ground Truth", "Predicted Image"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
        plt.axis("off")
    if savefig:
        plt.savefig(f"output_geometry_re\\train_pix2pix{step}.png")
    else:
        plt.show()
    plt.clf()
    plt.hist(np.array(tar[0, :, :]).flatten(), bins=100, alpha=0.5, label="Ground Truth")
    plt.hist(np.array(prediction[0, :, :, 0]).flatten(), bins=100, alpha=0.5, label="Prediction")
    plt.legend(loc="upper right")
    plt.title(f"Results at step {step}")
    if savefig:
        plt.savefig(f"output_geometry_re\\train_pix2pix_dist{step}.png")
    else:
        plt.show()


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



def fit(train_ds, test_ds, validation_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    for step, value_to_unpack in train_ds.repeat().take(steps).enumerate():
        (input_image, target) = value_to_unpack
        if (step) % 1000 == 0:
            display.clear_output(wait=True)
            if step != 0:
                print(f"Time taken for 10 steps: {time.time() - start:.2f} sec\n")
            start = time.time()
            generate_images(generator, example_input, example_target, step=step)
            if validation_ds is not None:
                one_one_plot_validation(validation_ds, generator)
            print(f"Step: {step // 1000}k")
        train_step(input_image, target, step)
        # Training step
        if (step + 1) % 10 == 0:
            print(".", end="", flush=True)
            # Save (checkpoint) the model every 5k steps
        if (step + 1) % 1000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def one_one_plot_validation(validation_dataset, model):
    # validation
    dataset = [value for counter, value in enumerate(validation_dataset)]
    data_diff = [data[-1].numpy() - data[0].numpy() for data in dataset]
    indexes = [np.nonzero(np.any(data[0] != 0, axis=1))[0][0] for data in data_diff]
    predictions = [model(data[0], training=False) for data in dataset]
    plt.clf()
    for counter, ind in enumerate(indexes):
        plt.plot(dataset[counter][-1].numpy()[0, ind, :], predictions[counter][-1].numpy()[ind, :, 0], "o", label=f"Validation set {counter}")
    plt.plot([0,1], [0,1], label="1-1 line")
    plt.xlabel("Expected normalized value")
    plt.ylabel("Predicted normalized value")
    plt.legend()
    plt.savefig("Validation_output.png")



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
    train_folder_cs = r"D:\GeoSchemaGen\tests\outputs\train\cs"
    train_folder_cptlike = r"D:\GeoSchemaGen\tests\outputs\train\cptlike"

    train_dataset, test_dataset, val_dataset = create_dataset(
        train_folder_cs, train_folder_cptlike, height, width, channels,
        BATCH_SIZE, TEST_PERCENTAGE)

    print(f"Fitting the model with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
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
AMAX = 4.5

train_folder_cs = r"D:\GeoSchemaGen\tests\outputs\train\cs"
train_folder_cptlike = r"D:\GeoSchemaGen\tests\outputs\train\cptlike"
height = 32
width = 512
channels = 1
BATCH_SIZE = 1
TEST_PERCENTAGE = 0.2  # 20% for testing
VAL_PERCENTAGE = 0.2   # 20% for validation

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