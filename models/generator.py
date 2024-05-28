import tensorflow as tf
import numpy as np


def encoder(filters: int, kernel: int, strides: tuple,
            batchnorm: bool = True, dropout: bool = False, dropout_rate: float = 0.5):
    """
    Encoder block to build the GAN generator

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


def decoder(filters: int, kernel: int, strides: int, batchnorm: bool = True, dropout: bool = True,
            dropout_rate: float = 0.5):
    """
    Decoder block to build the GAN generator

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


def Generator(input_shape: tuple = (256, 256, 4), base_filters: int = 64):
    # Calculate the number of layers based on the smallest dimension of the input shape
    # The number of layers is the log base 2 of the smallest dimension minus 1 (for the last layer)
    num_layers = int(np.log2(min(input_shape[:2]))) - 1

    # Define the input tensor to the generator
    generator_inputs = tf.keras.layers.Input(shape=input_shape)

    # This variable will hold the output of the current layer at each step
    current_layer_output = generator_inputs

    # Create the encoder/downsample stack
    encoder_layers = []
    for i in range(num_layers):
        # Double the number of filters with each layer (min 64, max 512)
        filters = base_filters * min(8, 2 ** i)
        # If the input is not square, use different strides to make it square
        if input_shape[0] != input_shape[1] and i == 0:
            # Add a encoder layer with different strides to the encoder stack
            encoder_layers.append(encoder(filters=filters, kernel=4, strides=(2, 1), batchnorm=False))
        else:
            # Add a encoder layer to the encoder stack
            encoder_layers.append(encoder(filters=filters, kernel=4, strides=(2, 2), batchnorm=(i != 0)))
    # Add the last encoder layer
    encoder_layers.append(encoder(filters=base_filters * 8, kernel=4, strides=(2, 2)))

    # Create the decoder/upsample stack
    decoder_layers = []
    for i in range(num_layers):
        # Halve the number of filters with each layer (min 64, max 512)
        filters = base_filters * min(8, 2 ** (num_layers - i - 1))
        # Add an decoder layer to the decoder stack
        decoder_layers.append(decoder(filters=filters, kernel=4, dropout=(i < 3)))

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
        OUTPUT_CHANNELS,
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
