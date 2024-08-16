import os
import time
import datetime
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from IPython import display
from models.generator import Generator_modular, generator_loss
from models.discriminator import Discriminator_modular, discriminator_loss
#from models.training import train_step, fit, generate_images, one_one_plot_validation
from utils.preprocessing import create_dataset

##### USER DEFINED PARAMETERS FOR SCHEMAGAN ############################################################################
BATCH_SIZE = 1
AMAX = 4.5
OUTPUT_CHANNELS = 1
LAMBDA = 100
IMAGE_SIZE = 256
IMAGE_IDX = 255

train_folder_cs = r"D:\GeoSchemaGen\tests\outputs\train\cs"
train_folder_cptlike = r"D:\GeoSchemaGen\tests\outputs\train\cptlike"
height = 32
width = 512
channels = 1
BATCH_SIZE = 1
TEST_PERCENTAGE = 0.2  # 20% for testing
VAL_PERCENTAGE = 0.2   # 20% for validation

########################################################################################################################



@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        print("Input Image Shape:", input_image.shape)
        print("Target Image Shape:", target.shape)
        print("step:", step)
        # Generate the output
        gen_output = generator(input_image, training=True)
        # Print the shape of the generated output
        tf.print("Generated Output Shape:", gen_output.shape)

        # Calculate the generator loss
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        # Print the shape of discriminator outputs
        tf.print("Discriminator Real Output Shape:", disc_real_output.shape)
        tf.print("Discriminator Generated Output Shape:", disc_generated_output.shape)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        # Calculate the discriminator loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Calculate the gradients for generator and discriminator
    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables)

    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables))

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


def set_up_and_train_2d_model():
    #directory_inputs = "D:/sheetpile/ai_model/inputs_geometry_re/"
    #train_dataset, test_dataset = set_up_data_as_input(directory_inputs)

    train_folder_cs = r"D:\GeoSchemaGen\tests\outputs\train\cs"
    train_folder_cptlike = r"D:\GeoSchemaGen\tests\outputs\train\cptlike"

    train_dataset, test_dataset, val_dataset = create_dataset(
        train_folder_cs, train_folder_cptlike, height, width, channels,
        BATCH_SIZE, TEST_PERCENTAGE)

    print(f"Fitting the model with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
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



if __name__ == "__main__":
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment


    # define optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    log_dir = "logs_geometry_re/"

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit_train_geometry_re/keep" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # We already have the loss object defined in the LOSS equation.
    #loss_object = tf.keras.losses.MeanSquaredError()

    print("Setting up the generator......")
    generator = Generator_modular(input_size=(32, 512), no_inputs=1)
    print(generator.summary())

    print("Setting up the discriminator......")
    discriminator = Discriminator_modular(input_size=(32, 512), no_inputs=1)
    print(discriminator.summary())

    checkpoint_dir = "D:\sheetpile\\ai_model\\training_checkpoints_2d_geometry_refined"
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
