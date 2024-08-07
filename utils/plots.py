from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


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


def plot_input_and_target_from_dataset(dataset: tf.data.Dataset, num_samples: int):
    """
    Plot input and target image pairs side-by-side from a TensorFlow dataset.

    Args:
        dataset (tf.data.Dataset): TensorFlow dataset containing input-target pairs.
        num_samples (int): Number of sample pairs to plot.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2))

    # Iterate over the dataset to fetch the images
    for i, (inputs, targets) in enumerate(dataset.take(num_samples)):
        if i >= num_samples:
            break
        # Plot the input image
        axes[i, 0].imshow(inputs[0, :, :, 0], cmap='viridis')
        axes[i, 0].set_title(f"Input {i+1}")
        axes[i, 0].axis('off')

        # Plot the target image
        axes[i, 1].imshow(targets[0, :, :, 0], cmap='viridis')
        axes[i, 1].set_title(f"Target {i+1}")
        axes[i, 1].axis('off')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()