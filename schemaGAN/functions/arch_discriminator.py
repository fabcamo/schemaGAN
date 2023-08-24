from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Concatenate, LeakyReLU
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam



def define_discriminator(image_shape):
    """
    Define the discriminator model for schemaGAN.

    Args:
        image_shape (tuple): The shape of the input images (height, width, channels).

    Returns:
        tensorflow.keras.models.Model: The compiled discriminator model.
    """

    # Weight initialization
    init = RandomNormal(stddev=0.02)  # Initialized with a standard deviation of 0.02 as in the original paper
    # Source image input
    in_src_image = Input(shape=image_shape)  # Input for the image we want to convert
    # Target image input
    in_target_image = Input(shape=image_shape)  # Input for the image we want to generate after training
    # Concatenate images, channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    # Define the discriminator architecture

    # C64: Convolutional layer with 64 filters, 4x4 kernel, and Stride 2x2
    d = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)

    # C128: Convolutional layer with 128 filters, 4x4 kernel, and Stride 1x2
    d = Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C256: Convolutional layer with 256 filters, 4x4 kernel, and Stride 1x2
    d = Conv2D(filters=256, kernel_size=(4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C512: Convolutional layer with 512 filters, 4x4 kernel, and Stride 1x2
    d = Conv2D(filters=512, kernel_size=(4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Another C512 layer: Convolutional layer with 512 filters, 4x4 kernel, and Stride 1x2
    d = Conv2D(filters=512, kernel_size=(4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Second last output layer: Convolutional layer with 512 filters, 4x4 kernel, and Stride 1x1
    d = Conv2D(filters=512, kernel_size=(4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Patch output
    d = Conv2D(filters=1, kernel_size=(4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    # Define the model
    model = Model([in_src_image, in_target_image], patch_out)

    # Compile the model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5], metrics=['mae', 'accuracy'])

    return model
