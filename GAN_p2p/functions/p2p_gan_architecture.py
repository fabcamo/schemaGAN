from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, Dropout,  BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # Make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False  # Descriminator layers set to untrainable in the combined GAN but
            # standalone descriminator will be trainable.

    # Define the source image (CPT-like)
    in_src = Input(shape=image_shape)
    # suppy the image as input to the generator
    gen_out = g_model(in_src)
    # supply the input image and generated image as inputs to the discriminator
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and disc. output as outputs
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)

    # IMPORTAT TO UNDERSTAND THISSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
    # Total loss is the weighted sum of adversarial loss (BCE) and L1 loss (MAE)
    # Authors suggested weighting BCE vs L1 as 1:100.
    model.compile(loss=['binary_crossentropy', 'mae'],  optimizer=opt, loss_weights=[1, 100])
    return model

