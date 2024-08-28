import tensorflow as tf
import numpy as np

from utils.layers import downsample, upsample

"""
This is a script to build the generator model for the GAN based on a pix2pix Generator U-Net architecture,
based on https://www.tensorflow.org/tutorials/generative/pix2pix and the work done by @EleniSmyrniou and @FabianCampos
"""

def Generator_modular(input_size: tuple = (256, 256), no_inputs: int = 4, OUTPUT_CHANNELS: int = 1,
                      base_filters: int = 64, dropout_layers: int = 3):
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

    # Calculate the number of layers based on both dimensions of the input shape as powers of 2 with the log2 function
    log2_width = int(np.log2(input_shape[0]))
    log2_height = int(np.log2(input_shape[1]))

    # Calculate the ratio between the width and height
    ratio = np.abs(log2_width - log2_height)
    print(f"Width: {log2_width}, Height: {log2_height}, Ratio: {ratio}")
    # Calculate the inverse ratio as the maximum number of layers minus the ratio to reach a square shape
    inverse_ratio = np.abs(max(log2_width, log2_height) - ratio)

    # Worked out number of layers needed to reach a dimension of 1x1 (2^0)
    no_layers = max(log2_width, log2_height)

    # Define the input tensor to the generator
    generator_inputs = tf.keras.layers.Input(shape=input_shape)

    # This variable will hold the output of the current layer at each step
    current_layer_output = generator_inputs

    # Create the encoder/downsample stack
    encoder_layers = []
    # Loop through the layers needed to reach a dimension of 1x1
    for i in range(no_layers):
        # If the shape is squared, use stride (2, 2) for all layers
        if input_shape[0] == input_shape[1]:
            strides = (2, 2)

        # If the shape is not squared, use stride (2, 1) or (1, 2) to reach a size of 2x2
        elif input_shape[0] != input_shape[1]:
            # Loop first through the short dimension to get it to size 2 with square strides
            if i < ratio:
                # Use either stride (2, 1) or (1, 2) to reach 2x2 depending on the shape
                if input_shape[0] > input_shape[1]:
                    strides = (2, 1)
                elif input_shape[0] < input_shape[1]:
                    strides = (1, 2)

            # After the shape is squared, we can use stride (2, 2) for all layers
            elif i >= ratio:
                strides = (2, 2)

        # Else, print an error message
        else:
            print("ERROR: in the encoder block")

        # Double the number of filters with each layer (min 64, max 512)
        filters = base_filters * min(8, 2 ** i)
        # Add an encoder layer to the encoder stack with batch normalization after the first layer
        encoder_layers.append(downsample(filters=filters, kernel=4, strides=strides, batchnorm=(i != 0)))

    # Create the decoder stack
    decoder_layers = []
    # Calculate the number of steps needed in reverse to reach the original irregular shape
    for i in range(no_layers):
        # If the shape is squared, use stride (2, 2) for all layers
        if input_shape[0] == input_shape[1]:
            strides = (2, 2)

        # If the shape is not squared, first scale squared with stride (2, 2)
        elif input_shape[0] != input_shape[1]:
            # Loop first through the short dimension to get it to size 2 with square strides
            if i < inverse_ratio:
                strides = (2, 2)

            # After we reach the original size in the short dimension, we can use stride (2, 1) or (1, 2) to reach the original size
            elif i >= inverse_ratio:
                # Use either stride (2, 1) or (1, 2) to reach 2x2 depending on the shape
                if input_shape[0] > input_shape[1]:
                    strides = (2, 1)
                elif input_shape[0] < input_shape[1]:
                    strides = (1, 2)

        # Calculate the number of filters for the current layer
        if i < no_layers - 4:
            filters = base_filters * 8
        else:
            filters = base_filters * min(8, 2 ** (no_layers - i - 2))

        # Add a decoder layer to the decoder stack
        decoder_layers.append(
            upsample(filters=filters, kernel=4, strides=strides, batchnorm=True, dropout=(i < dropout_layers)))

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

    # Check the last layer and adjust the stride accordingly
    # If it is squared, use stride (2, 2)
    if input_shape[0] == input_shape[1]:
        strides = (2, 2)
    # If it is not squared, use stride (2, 1) to square it according to the smaller dimension
    elif input_shape[0] > input_shape[1]:
        strides = (2, 1)
    # If it is not squared, use stride (1, 2) to square it according to the smaller dimension
    elif input_shape[0] < input_shape[1]:
        strides = (1, 2)
    # Else, print an error message
    else:
        print("Error in the input shape, please provide a valid shape")

    # Add the last layer
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    last_layer = tf.keras.layers.Conv2DTranspose(
        filters=OUTPUT_CHANNELS,
        kernel_size=4,
        strides=strides,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )
    # Pass the current output through the last layer to get the final output
    final_output = last_layer(current_layer_output)

    # Check if the input and output shapes are the same
    if input_shape[0] != final_output.shape[1] or input_shape[1] != final_output.shape[2]:
        print("Input and output shapes do not match, please check the input shape and the number of layers")
    else:
        print("Input and output shapes match")

    # Return the generator model
    return tf.keras.Model(inputs=generator_inputs, outputs=final_output)



def generator_loss(disc_generated_output: tf.Tensor, gen_output: tf.Tensor, target: tf.Tensor,
                   loss_object: tf.keras.losses = tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   LAMBDA: int = 100):
    """
    Calculate the generator loss for the GAN based on the discriminator output, the generator output, the target,
    the loss object, and the lambda value for the L1 loss. The generator loss is the sum of the adversarial loss
    and the L1 loss. The adversarial loss is the binary cross-entropy loss between the discriminator output and
    a tensor of ones, while the L1 loss is the mean absolute error between the generator output and the target.

    Args:
        disc_generated_output (tf.Tensor): The output of the discriminator for the generated image.
        gen_output (tf.Tensor): The output of the generator.
        target (tf.Tensor): The target image.
        loss_object (tf.keras.losses): The loss object for the adversarial loss. Default is BinaryCrossentropy.
        LAMBDA (int): The lambda value for the L1 loss. Default is 100 as in the original pix2pix paper.

    Returns:
        tf.Tensor: The total generator loss which is the sum of the adversarial loss and the L1 loss.
        tf.Tensor: The adversarial loss which is the BCE loss between the discriminator output and a tensor of ones.
        tf.Tensor: The L1 loss which is the MAE between the generator output and the target.

    """
    # Compute the adversarial loss between the discriminator output and a tensor of ones (real image)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Compute the L1 loss between the generated image and the target image
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    # Total generator loss is the sum of the adversarial loss and the L1 loss
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


