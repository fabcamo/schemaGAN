import tensorflow as tf
import numpy as np

"""
This is a script to build the generator model for the GAN based on a pix2pix Generator U-Net architecture,
based on https://www.tensorflow.org/tutorials/generative/pix2pix and the work done by @EleniSmyrniou and @FabianCampos
"""

def downsample(filters: int, kernel: int, strides: tuple,
               batchnorm: bool = True, dropout: bool = False, dropout_rate: float = 0.5):
    """
    Downsample (encoder) block to build the GAN generator

    Args:
        filters (int): The number of filters in the convolutional layer.
        kernel (int): The size of the kernel.
        strides (tuple): The stride of the convolution.
        batchnorm (bool, optional): Whether to use batch normalization. Default is True.
        dropout (bool, optional): Whether to use dropout. Default is False.
        dropout_rate (float, optional): The dropout rate. Default is 0.5.

    Returns:
        result (tf.keras.Sequential): The output tensor of the encoder block.
    """
    # Weight initialization
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    # Create the encoder/downsampling block
    result = tf.keras.Sequential()
    # Add the encoder/downsampling layer
    result.add(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    # Conditionally add batch normalization
    if batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    # Conditionally add dropout
    if dropout:
        result.add(tf.keras.layers.Dropout(dropout_rate))

    # Leaky ReLU activation
    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters: int, kernel: int, strides: tuple, batchnorm: bool = True, dropout: bool = True,
            dropout_rate: float = 0.5):
    """

    Upsample (decoder) block to build the GAN generator

    Args:
        filters (int): The number of filters in the convolutional layer.
        kernel (int): The size of the kernel.
        strides (tuple): The stride of the convolution.
        batchnorm (bool, optional): Whether to use batch normalization. Default is True.
        dropout (bool, optional): Whether to use dropout. Default is True.
        dropout_rate (float, optional): The dropout rate. Default is 0.5.

    Returns:
        result (tf.keras.Sequential): The output tensor of the decoder block.
    """
    # Weight initialization
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    # Create the decoder/upsampling block
    result = tf.keras.Sequential()
    # Add the decoder/upsampling layer
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    # Conditionally add batch normalization
    if batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    # Conditionally add dropout
    if dropout:
        result.add(tf.keras.layers.Dropout(dropout_rate))

    # ReLU activation
    result.add(tf.keras.layers.ReLU())

    return result


def Generator(input_size: tuple = (256, 256), no_inputs: int = 4, OUTPUT_CHANNELS: int = 1, base_filters: int = 64,
              dropout_layers: int = 3):
    """
    Build the generator model for the GAN based on a pix2pix Generator U-Net architecture.
    Design decisions are based on the original pix2pix paper: https://arxiv.org/abs/1611.07004.
    Change the code to modify layer with or w.o. batch normalization, dropout, etc.

    -----> WORKS FOR SHAPES THAT ARE POWERS OF 2 <------

    Args:
        input_size (tuple, optional): The shape of the input tensor. Default is (256, 256)
        no_inputs (int, optional): The number of input channels. Default is 4, RGB uses 3.
        OUTPUT_CHANNELS (int, optional): The number of output channels. Default is 1, RGB uses 3.
        base_filters (int, optional): The number of filters in the first layer. Default is 64.
        dropout_layers (int, optional): The number of layers to apply dropout. Default is 3 based on pix2pix.

    Returns:
        tf.keras.Model: The generator model.
    """
    # Create the input_shape
    input_shape = (*input_size, no_inputs)

    # Calculate the number of layers based on both dimensions of the input shape
    num_layers_width = int(np.log2(input_shape[0]))
    num_layers_height = int(np.log2(input_shape[1]))
    # Assign the number of layers to a min a max dimension, needed for irregular shapes
    big_dimension = max(num_layers_width, num_layers_height)
    short_dimension = min(num_layers_width, num_layers_height)

    # Define the input tensor to the generator
    generator_inputs = tf.keras.layers.Input(shape=input_shape)

    # This variable will hold the output of the current layer at each step
    current_layer_output = generator_inputs

    # Create the encoder/downsample stack
    encoder_layers = []
    for i in range(big_dimension):
        # Double the number of filters with each layer (min 64, max 512)
        filters = base_filters * min(8, 2 ** i)
        # Adjust the strides to make the output square
        if i < short_dimension - 1 and short_dimension != big_dimension:
            strides = (1, 2)
        elif i >= short_dimension - 1:
            strides = (2, 2)
        else:
            strides = (2, 2)
        # Add a encoder layer to the encoder stack with batch normalization after the first layer
        encoder_layers.append(downsample(filters=filters, kernel=4, strides=strides, batchnorm=(i != 0)))

    # Create the decoder stack
    decoder_layers = []
    for i in range(big_dimension):
        # Calculate the number of filters for the current layer
        if i < big_dimension - 4:
            filters = base_filters * 8
        else:
            filters = base_filters * min(8, 2 ** (big_dimension - i - 2))
        # Adjust the strides to make the output square
        if i <= short_dimension - 1:
            strides = (2, 2)
        elif i > short_dimension - 1 and short_dimension != big_dimension:
            strides = (1, 2)
        else:
            strides = (2, 2)
        # Add a decoder layer to the decoder stack
        decoder_layers.append(upsample(filters=filters, kernel=4, strides=strides, batchnorm=True, dropout=(i < dropout_layers)))

    # This list will hold the output tensors that will be used for skip connections
    skip_connections = []
    for encoder_layer in encoder_layers:
        # Pass the current output through the encoder layer and update it
        current_layer_output = encoder_layer(current_layer_output)
        # Add the current output to the list of skip connections
        skip_connections.append(current_layer_output)

    # Reverse the skip connections to match the order of the decoder layers
    skip_connections = reversed(skip_connections[:-1])

    # Decoding and establishing the skip connections
    for decoder_layer, skip_connection in zip(decoder_layers, skip_connections):
        # Pass the current output through the decoder layer and update it
        current_layer_output = decoder_layer(current_layer_output)
        # Concatenate the current output with the corresponding skip connection
        current_layer_output = tf.keras.layers.Concatenate()([current_layer_output, skip_connection])

    # Add the last layer
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    last_layer = tf.keras.layers.Conv2DTranspose(
        filters=OUTPUT_CHANNELS,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )
    # Pass the current output through the last layer to get the final output
    final_output = last_layer(current_layer_output)

    # Return the generator model
    return tf.keras.Model(inputs=generator_inputs, outputs=final_output)


def Discriminator(input_shape: tuple = (256,256,4)):

    # Define the input tensor to the discriminator
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=input_shape, name='target_image')

