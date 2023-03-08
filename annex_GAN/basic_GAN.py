import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import zeros
from numpy.random import randn

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, MaxPooling2D, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import InputLayer, Reshape, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Check if GPUs are available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Import the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
all_images = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Check the image size
print('The shape of a single image is: ',all_images[0].shape)

# Plot to check the data
fig, axes = plt.subplots(1, 6, figsize=(10, 3))
for i in range(6):
    axes[i].imshow(all_images[i], cmap='Greys')
    axes[i].axis('off')
plt.show()

# Normalize the data between 0 - 1
all_images = all_images.astype("float32") / 255.0
# Because it is between 0-1> we are going to use SIGMOID activation


# We need to add an additional dimension to make it know how many channels are there
all_images = all_images.reshape(70000,28,28,1)
input_shape = all_images[0].shape


# HYPER-PARAMETERS ####################################################################################################
no_channels = 1         # number of channels in the image
latent_dim = 128        # user defined number as input to the generator
batch_size = 32         # user defined batch size

n_samples = 1


#### DEFINE THE FUNCTIONS #############################################################################################


# Discriminator architecture model
def discriminator_model(input_shape):
    model = Sequential()
    # Layer to be used as an entry point into a Network
    model.add(InputLayer(input_shape=input_shape))
    # Add convolutional layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Add Max Pooling layer
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    # Flatten the 2D input
    model.add(Flatten())
    # Drop 40% of the data
    model.add(Dropout(0.4))
    # Add the Dense layers
    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=0.2))
    # Add FINAL fake or real layer
    model.add(Dense(units=1, activation='sigmoid'))

    # Define the optimizer
    opt = Adam(lr=0.0002, beta_1=0.5)
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acuracy'])

    return model


#test_discr = discriminator_model(input_shape)
#print(test_discr.summary())



# Generator architecture model
def generator_model(latent_dim):
    model = Sequential()

    # Define initial geometry vector in order to, after all operations
    # arrive at a dimension that is equal to the input image
    n_nodes = 7 * 7 * 128

    # Layer to be used as an entry point into a Network
    model.add(InputLayer(input_shape=latent_dim))
    # Add Dense layer starting from a given number of neurons
    model.add(Dense(units=n_nodes))
    model.add(LeakyReLU(alpha=0.2))

    # Reshape from 1D to 2D to apply Conv2D
    model.add(Reshape((7, 7, 128)))

    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))  # 16x16x128
    model.add(LeakyReLU(alpha=0.2))


    # generate
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'))  # 32x32x128
    model.add(LeakyReLU(alpha=0.2))
    # I DON'T UNDERSTAND WHAT IS GOING ON HERE, really....
    # The size (7x7) has the match the input for the no. Parameters to match the initial no. parameters....
    model.add(Conv2D(filters=1, kernel_size=(7, 7), activation='sigmoid', padding='same'))  # 32x32x3

    #NOTE> MODEL NOT COMPILED
    return model  # Model not compiled as it is not directly trained like the discriminator.
    # Generator is trained via GAN combined model.

test_gen = generator_model(128)
print(test_gen.summary())



# Define the combined generator-discriminator model
# This is done to train the generator, while keeping the discriminator constant
# The discriminator is trained separately
def gan_model(generator, discriminator):
    # Make the discriminator non-trainable
    discriminator.trainable = False

    # Initiate the Sequential model
    model.Sequential()
    # Add the discriminator and generator
    model.add(generator)
    model.add(discriminator)
    # Compile the model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model



# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
 # generate points in the latent space
 x_input = randn(latent_dim * n_samples)
 # reshape into a batch of inputs for the network
 x_input = x_input.reshape(n_samples, latent_dim)
 return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
 # generate points in latent space
 x_input = generate_latent_points(latent_dim, n_samples)
 # predict outputs
 X = g_model.predict(x_input)
 # create 'fake' class labels (0)
 y = zeros((n_samples, 1))
 return X, y

g_model = generator_model(latent_dim)

fake_image = generate_fake_samples(g_model, latent_dim, n_samples)
print('the shape of the fake image is: ', fake_image[0].shape)

fake = fake_image[0]
fake = np.reshape(fake, (28, 28))
print('the shape of the fake image after processing is: ', fake.shape)

#print(fake)
plt.imshow(fake)
plt.show()