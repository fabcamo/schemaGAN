import tensorflow as tf
import numpy as np

from models.layers import downsample


def Discriminator_modular(input_size: tuple = (256, 256), no_inputs: int = 4, base_filters: int = 64):
    """
    Build the discriminator model for the GAN based on a PatchGAN architecture.
    This model is modular and can adapt to different input shapes and number of input channels.

    Args:
        input_size (tuple, optional): The shape of the input tensor. Default is (256, 256).
        no_inputs (int, optional): The number of input channels. Default is 4.
        base_filters (int, optional): The number of filters in the first layer. Default is 64.

    Returns:
        tf.keras.Model: The discriminator model.
    """
    # Define the input tensor to the discriminator
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    # Create the input_shape and target_shape. The target shape is the same as the input shape, but with only 1 channel
    # because the discriminator only needs to classify the real/fake image.
    input_shape = (*input_size, no_inputs)
    target_shape = (*input_size, 1)

    # Define the input and target tensors
    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=target_shape, name='target_image')

    # Concatenate the input and target tensors
    x = tf.keras.layers.concatenate([inp, tar]) # (batch_size, 256, 256, no_inputs+1)

    # Define the discriminator architecture
    # Calculate the number of layers based on both dimensions of the input shape
    num_layers_width = int(np.log2(input_size[0]))
    num_layers_height = int(np.log2(input_size[1]))
    # Assign the number of layers to a min and max dimension, needed for irregular shapes
    big_dimension = max(num_layers_width, num_layers_height)
    short_dimension = min(num_layers_width, num_layers_height)

    # Calculate the number of downsample layers as the big_dimension - 5 in order to have a 32x32 output
    # before going into the final convolutional layers
    num_downsample_layers = big_dimension - 5

    # Create the downsample stack
    down_layers = []
    for i in range(big_dimension - num_downsample_layers):
        # Double the number of filters with each layer (min 64, max 512)
        filters = base_filters * min(8, 2 ** i)
        # Adjust the strides to make the output square
        if i < short_dimension - 1 and short_dimension != big_dimension:
            strides = (1, 2)
        elif i >= short_dimension - 1:
            strides = (2, 2)
        else:
            strides = (2, 2)
        # Add a downsample layer to the stack with batch normalization after the first layer
        down_layers.append(downsample(filters=filters, kernel=4, strides=strides, batchnorm=(i != 0)))

    # Apply the downsample layers
    current_layer_output = x
    for layer in down_layers:
        current_layer_output = layer(current_layer_output)

    # Define the last layer of the discriminator model
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(current_layer_output)
    # The last layer does not have batch normalization and uses a linear activation function
    conv = tf.keras.layers.Conv2D(base_filters * 8,
                                  kernel_size=4,
                                  strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)

    # Apply batch normalization and leaky ReLU activation function
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    # Zero padding and the last convolutional layer with a single filter
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    # The output of the discriminator is a single value
    last = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, kernel_initializer=initializer)(zero_pad2)

    # Return the discriminator model with the input and target tensors as inputs and the output tensor as output
    return tf.keras.Model(inputs=[inp, tar], outputs=last)