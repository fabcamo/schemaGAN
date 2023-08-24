from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, Dropout,  BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def define_gan(g_model, d_model, image_shape):
    """
    Define the combined generator and discriminator model for updating the generator.

    Args:
        g_model (tensorflow.keras.Model): The generator model.
        d_model (tensorflow.keras.Model): The discriminator model.
        image_shape (tuple): The shape of the input images.

    Returns:
        tensorflow.keras.Model: The compiled GAN model.
    """

    # Make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False  # Discriminator layers set to untrainable in the combined GAN
            # Standalone discriminator will be trainable.

    # Define the source image (CPT-like)
    in_src = Input(shape=image_shape)
    # Supply the image as input to the generator
    gen_out = g_model(in_src)
    # Supply the input image and generated image as inputs to the discriminator
    dis_out = d_model([in_src, gen_out])
    # Source image as input, generated image, and discriminator output as outputs
    model = Model(in_src, [dis_out, gen_out])

    # Compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)

    # The total loss is the weighted sum of adversarial loss (BCE) and L1 loss (MAE).
    # Authors suggested weighting BCE vs L1 as 1:100.
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])

    return model