from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Concatenate, LeakyReLU
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam


def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)  # As described in the original paper

    # source image input
    in_src_image = Input(shape=image_shape)  # Image we want to convert to another image
    # target image input
    in_target_image = Input(shape=image_shape)  # Image we want to generate after training.

    # concatenate images, channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    # C64: 4x4 kernel Stride 2x2
    d = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128: 4x4 kernel Stride 2x2
    d = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256: 4x4 kernel Stride 2x2
    d = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512: 4x4 kernel Stride 2x2
    # Not in the original paper. Comment this block if you want.
    d = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer : 4x4 kernel but Stride 1x1
    d = Conv2D(filters=512, kernel_size=(4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(filters=1, kernel_size=(4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    # The model is trained with a batch size of one image and Adam opt.
    # with a small learning rate and 0.5 beta.
    # The loss for the discriminator is weighted by 50% for each model update.

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5], metrics=['mean_squared_error'])
    return model



def define_discriminator_512x32(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)  # As described in the original paper

    # source image input
    in_src_image = Input(shape=image_shape)  # Image we want to convert to another image
    # target image input
    in_target_image = Input(shape=image_shape)  # Image we want to generate after training.

    # concatenate images, channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    # Starting from 32x512

    # C64: 4x4 kernel Stride 2x2> 16x256
    d = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128: 4x4 kernel Stride 1x2> 16x128
    d = Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256: 4x4 kernel Stride 1x2> 16x64
    d = Conv2D(filters=256, kernel_size=(4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512: 4x4 kernel Stride 1x2> 16x32
    d = Conv2D(filters=512, kernel_size=(4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512: 4x4 kernel Stride 1x2> 16x16
    d = Conv2D(filters=512, kernel_size=(4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer : 4x4 kernel but Stride 1x1> 16x16
    d = Conv2D(filters=512, kernel_size=(4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(filters=1, kernel_size=(4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    # The model is trained with a batch size of one image and Adam opt.
    # with a small learning rate and 0.5 beta.
    # The loss for the discriminator is weighted by 50% for each model update.

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5], metrics=['mean_squared_error'])
    return model
