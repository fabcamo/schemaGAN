import tensorflow as tf
import time

#from IPython import display
from utils.plots import generate_images
from models.generator import Generator_modular, generator_loss
from models.discriminator import Discriminator_modular, discriminator_loss
from utils.plots import one_one_plot_validation

@tf.function
def train_step(input_image: tf.Tensor, target: tf.Tensor, step):
    """
    Perform a single training step for the generator and discriminator. It calculates the generator loss and
    discriminator loss based on the input image, the target image, and the current step. The function uses a
    gradient tape to calculate the gradients for both the generator and discriminator, applies the gradients to
    the optimizer, and writes the generator loss and discriminator loss to TensorBoard.

    Args:
        input_image (tf.Tensor): The input image for the generator.
        target (tf.Tensor): The target image for the generator.
        step (int): The current step in the training process.

    Returns:
        gen_total_loss (tf.Tensor): The total generator loss for the GAN.
        disc_loss (tf.Tensor): The discriminator loss for the GAN.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate the output image
        gen_output = generator(input_image, training=True)

        # Pass the generated image to the discriminator and get the output for both real and generated images
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # Calculate the generator loss based on the discriminator output, the generator output, and the target
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)

        # Calculate the discriminator loss based on the discriminator output for real and generated images
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # Calculate the gradients for generator and discriminator
        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # Apply the gradients to the optimizer
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        # Write the generator loss and discriminator loss to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)

        return gen_total_loss, disc_loss


def fit(train_ds: tf.data.Dataset, test_ds: tf.data.Dataset, validation_ds: tf.data.Dataset = None, steps: int = 5000):
    """
    Train the generator and discriminator models using the training dataset and the test dataset for a specified number
    of steps. The function generates and displays images using the generator model and the example input and target
    images from the test dataset. It also saves the model every 5000 steps to preserve the training progress.

    Args:
        train_ds (tf.data.Dataset): The training dataset.
        test_ds (tf.data.Dataset): The test dataset.
        validation_ds (tf.data.Dataset, optional): The validation dataset. Default is None.
        steps (int): The number of steps to train the models. Default is 5000.

    Returns:
        None
    """

    # Get an example batch of input and target images from the test dataset for visualization
    example_input, example_target = next(iter(test_ds.take(1)))

    # Record the start time to measure the duration of the training intervals
    start = time.time()

    # Iterate over the training dataset for the specified number of steps
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        # Every 1000 steps, perform the following:
        if (step) % 1000 ==0:
            # Clear the current output in the terminal to make the display cleaner
            display.clear_output(wait=True)

        # If not the first step, print the time taken for the last 1000 steps
        if step != 0:
            print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
        # Reset the start time for the next interval
        start = time.time()
        # Generate and display images using the generator model and the example input and target images
        generate_images(generator, example_input, example_target)
        # Print the step number
        print(f"step: {step//1000}k")

        # If a validation dataset is provided, generate and display images using the generator model and the example
        if validation_ds is not None:
            one_one_plot_validation(validation_ds, generator)
        print(f"Step: {step // 1000}k")

    # Perform a single training step for the generator and discriminator, updating the model
    train_step(input_image=input_image, target=target, step=step)

    # Print a dot to indicate progress every 10 steps
    if (step +1) % 10 == 0:
        print('.', end='', flush=True)

    # Save (checkpoint) the model every 5000 steps to preserve the training progress
    if (step + 1) % 5000 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)


