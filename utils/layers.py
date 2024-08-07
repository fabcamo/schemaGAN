import tensorflow as tf


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

