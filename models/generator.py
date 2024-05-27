import tensorflow as tf





def encoder(filters: int, kernel: int, stride: int, batchnorm: bool = True, dropout: bool = False):
    """
    Encoder block to build the GAN generator

    Args:
        filters (int): The number of filters in the convolutional layer.
        kernel (int): The size of the kernel.
        stride (int): The stride of the convolution.
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
    # Conditionally add dropout
    if dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    # Leaky ReLU activation
    result.add(tf.keras.layers.LeakyReLU())

    return result


def decoder(filters: int, kernel: int, stride: int, batchnorm: bool = True, dropout: bool = True, skip_tensor = None):
    """
    Decoder block to build the GAN generator

    Args:
        filters (int): The number of filters in the convolutional layer.
        kernel (int): The size of the kernel.
        stride (int): The stride of the convolution.
        batchnorm (bool, optional): Whether to use batch normalization. Default is True.
        dropout (bool, optional): Whether to use dropout. Default is True.
        skip_tensor (tf.Tensor, optional): The tensor to concatenate as a skip connection. Default is None.

    Returns:
        result (tf.keras.Sequential): The output tensor of the decoder block.
    """
    # Weight initialization
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    # Create the decoder block
    result = tf.keras.Sequential()
    # Add the upsampling layer
    result.add(
        tf.keras.layers.Conv2DTranspose(
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
    # Conditionally add dropout
    if dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    # Add skip connection if skip_tensor is provided
    if skip_tensor is not None:
        result.add(tf.keras.layers.Concatenate())
        result.add(skip_tensor)

    # ReLU activation
    result.add(tf.keras.layers.ReLU())

    return result