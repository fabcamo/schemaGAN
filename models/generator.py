import tensorflow as tf





def encoder(filters: int, kernel: int, stride: int, batchnorm: bool = True):
    """
    Encoder block to build the GAN generator

    Args:
        filters (int): The number of filters in the convolutional layer.
        size (int): The size of the kernel.
        batchnorm (bool, optional): Whether to use batch normalization. Default is True.

    Returns:
        result (tf.keras.Sequential): The output tensor of the encoder block.
    """
    # Weight initialization
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    # Create the encoder block
    result = tf.keras.Sequential()
    # Add the downsampling layer
    result.add(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=stride,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    # Conditionally add batch normalization
    if batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    # Leaky ReLU activation
    result.add(tf.keras.layers.LeakyReLU())

    return result


def decoder(filters: int, kernel: int, stride: int, dropout: bool = True):