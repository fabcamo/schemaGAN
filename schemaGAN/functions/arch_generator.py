from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, LeakyReLU
from tensorflow.keras.layers import Activation, Dropout,  BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal



def regular_encoder_block(layer_in, n_filters, batchnorm=True):
    """
    Create a regular encoder block for a neural network.

    Args:
        layer_in (tensorflow.Tensor): The input tensor to the encoder block.
        n_filters (int): The number of filters in the convolutional layer.
        batchnorm (bool, optional): Whether to use batch normalization. Default is True.

    Returns:
        tensorflow.Tensor: The output tensor of the encoder block.
    """

    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # Add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # Conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # Leaky ReLU activation
    g = LeakyReLU(alpha=0.2)(g)

    return g



def irregular_encoder_block(layer_in, n_filters, batchnorm=True):
    """
    Create an irregular encoder block for a neural network.

    Args:
        layer_in (tensorflow.Tensor): The input tensor to the encoder block.
        n_filters (int): The number of filters in the convolutional layer.
        batchnorm (bool, optional): Whether to use batch normalization. Default is True.

    Returns:
        tensorflow.Tensor: The output tensor of the encoder block.
    """

    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # Add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(layer_in)
    # Conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # Leaky ReLU activation
    g = LeakyReLU(alpha=0.2)(g)

    return g



def regular_decoder_block(layer_in, skip_in, n_filters, dropout=True):
    """
    Create a regular decoder block for a neural network generator.

    Args:
        layer_in (tensorflow.Tensor): The input tensor to the decoder block.
        skip_in (tensorflow.Tensor): The skip connection tensor to concatenate.
        n_filters (int): The number of filters in the transposed convolutional layer.
        dropout (bool, optional): Whether to use dropout. Default is True.

    Returns:
        tensorflow.Tensor: The output tensor of the decoder block.
    """

    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # Add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # Add batch normalization
    g = BatchNormalization()(g, training=True)
    # Conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # Merge with skip connection
    g = Concatenate()([g, skip_in])
    # ReLU activation
    g = Activation('relu')(g)

    return g



def irregular_decoder_block(layer_in, skip_in, n_filters, dropout=True):
    """
    Create an irregular decoder block for a neural network generator.

    Args:
        layer_in (tensorflow.Tensor): The input tensor to the decoder block.
        skip_in (tensorflow.Tensor): The skip connection tensor to concatenate.
        n_filters (int): The number of filters in the transposed convolutional layer.
        dropout (bool, optional): Whether to use dropout. Default is True.

    Returns:
        tensorflow.Tensor: The output tensor of the decoder block.
    """

    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # Add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(layer_in)
    # Add batch normalization
    g = BatchNormalization()(g, training=True)
    # Conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # Merge with skip connection
    g = Concatenate()([g, skip_in])
    # ReLU activation
    g = Activation('relu')(g)

    return g



def define_generator(image_shape=(32, 512, 1)):
    """
    Define a U-net style generator model for a neural network.

    Args:
        image_shape (tuple, optional): The shape of the input images (height, width, channels). Default is (32, 512, 1).

    Returns:
        tensorflow.keras.models.Model: The compiled generator model.
    """

    # Weight initialization
    init = RandomNormal(stddev=0.02)

    # Image input
    in_image = Input(shape=image_shape)

    # Encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = regular_encoder_block(in_image, 64, batchnorm=False)
    e2 = regular_encoder_block(e1, 128)
    e3 = regular_encoder_block(e2, 256)
    e4 = regular_encoder_block(e3, 512)
    e5 = irregular_encoder_block(e4, 512)
    e6 = irregular_encoder_block(e5, 512)
    e7 = irregular_encoder_block(e6, 512)
    e8 = irregular_encoder_block(e7, 512)

    # Bottleneck, no batch norm and ReLU
    b = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e8)
    b = Activation('relu')(b)

    # Decoder model: CD512-CD512-CD512-C512-C256-C128-C64
    d1 = regular_decoder_block(b, e8, 512)
    d2 = irregular_decoder_block(d1, e7, 512)
    d3 = irregular_decoder_block(d2, e6, 512)
    d4 = irregular_decoder_block(d3, e5, 512)
    d5 = irregular_decoder_block(d4, e4, 512, dropout=False)
    d6 = regular_decoder_block(d5, e3, 256, dropout=False)
    d7 = regular_decoder_block(d6, e2, 128, dropout=False)
    d8 = regular_decoder_block(d7, e1, 64, dropout=False)

    # Output
    g = Conv2DTranspose(image_shape[2], (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d8)
    out_image = Activation('tanh')(g)

    # Define model
    model = Model(in_image, out_image)

    return model