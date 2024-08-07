import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.training import train_step, fit


def set_up_and_train_model():
    """
    Set up and train the schemaGAN model. The function loads the data from the pickle files, sets up the training data,
    and trains the model. The function also generates and displays images using the generator model and the example
    input and target images from the test dataset.

    """
    # Parse the input data
    directory_inputs = "D:/sheetpile/ai_model/inputs_geometry_re/"
    # Load the data from the pickle files into a numpy array for training and testing
    train_dataset, test_dataset = set_up_data_as_input(directory_inputs)

    # Train the model using the training dataset and test dataset; NO validation dataset
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


def load_from_pickle(file):
    model = pickle.load(open(file, "rb"))
    feature_names = model['feature_names']
    all_data = model['all_data']
    return feature_names, np.array(all_data).astype(np.float32)

def _input_fn(inputs, target):
    input_dataset = tf.convert_to_tensor(tf.constant(inputs))
    train_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset)
    train_input_dataset = train_input_dataset.batch(BATCH_SIZE)
    target_dataset = tf.convert_to_tensor(tf.constant(target))
    train_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset)
    train_target_dataset = train_target_dataset.batch(BATCH_SIZE)
    train_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))
    return train_dataset


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