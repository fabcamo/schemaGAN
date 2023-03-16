import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.layers import InputLayer, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Import data from MNIST dataset
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
plt.show()      # To show the image
plt.clf()       # To clear the memory



# Plot to check the data
fig, axes = plt.subplots(1, 6, figsize=(10, 3))
for i in range(6):
    axes[i].imshow(all_images[i], cmap='Greys')
    axes[i].axis('off')
#plt.show()

# Normalize the data between 0 - 1
all_images = all_images.astype("float32") / 255.0
# Because it is between 0-1> we are going to use SIGMOID activation


# We need to add an additional dimension to make it know how many channels are there
all_images = all_images.reshape(70000,28,28,1)
input_shape = all_images[0].shape


# HYPER-PARAMETERS ####################################################################################################
no_channels = 1         # number of channels in the image
latent_dim = 100        # user defined number as input to the generator
batch_size = 32         # user defined batch size

n_samples = 1


#### DEFINE THE FUNCTIONS #############################################################################################


# Discriminator architecture model
def discriminator_model(input_shape):
    model = Sequential()
    # Layer to be used as an entry point into a Network
    model.add(InputLayer(input_shape=input_shape))
    # Flatten the 2D input
    model.add(Flatten())
    # Add the Dense layers
    model.add(Dense(units=1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=0.2))
    # Add FINAL fake or real layer
    model.add(Dense(units=1, activation='sigmoid'))

    # Define the optimizer
    opt = Adam(lr=0.0002, beta_1=0.5)
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model




# Generator architecture model
def generator_model(latent_dim):
    n_nodes = 28 * 28  # because the domension of the output pictures should be 28x28

    model = Sequential()
    # Layer to be used as an entry point into a Network
    model.add(InputLayer(input_shape=latent_dim))
    # Add Dense layer starting from a given number of neurons
    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=n_nodes))
    model.add(LeakyReLU(alpha=0.2))
    # Reshape from 1D to 2D to apply Conv2D
    model.add(Reshape((28, 28)))

    #NOTE> MODEL NOT COMPILED
    return model  # Model not compiled as it is not directly trained like the discriminator.
    # Generator is trained via GAN combined model.

'''
# Define the combined generator-discriminator model
# This is done to train the generator, while keeping the discriminator constant
# The discriminator is trained separately
def gan_model(generator, discriminator):
    # Make the discriminator non-trainable
    discriminator.trainable = False

    # Initiate the Sequential model
    model = Sequential()
    # Add the discriminator and generator
    model.add(generator)
    model.add(discriminator)
    # Compile the model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model
'''

generator = generator_model(latent_dim)
discriminator = discriminator_model(input_shape)

GAN = Sequential([generator, discriminator])
discriminator.trainable = False
GAN.compile(loss="binary_crossentropy", optimizer="adam")




##### TRAIN THE SIMPLE GAN ##################################################################

my_data = x_train
dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
epochs = 4



generator, discriminator = GAN.layers


# For every epcoh
for epoch in range(epochs):
    print(f"Currently on Epoch {epoch + 1}")
    i = 0
    # For every batch in the dataset
    for X_batch in dataset:
        i = i + 1
        if i % 100 == 0:
            print(f"\tCurrently on batch number {i} of {len(my_data) // batch_size}")
        #####################################
        ## TRAINING THE DISCRIMINATOR ######
        ###################################

        # Create Noise
        noise = tf.random.normal(shape=[batch_size, latent_dim])

        # Generate numbers based just on noise input
        gen_images = generator(noise)

        # Concatenate Generated Images against the Real Ones
        # TO use tf.concat, the data types must match!
        X_fake_vs_real = tf.concat([gen_images, tf.dtypes.cast(X_batch, tf.float32)], axis=0)

        # Targets set to zero for fake images and 1 for real images
        y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)

        # This gets rid of a Keras warning
        discriminator.trainable = True

        # Train the discriminator on this batch
        discriminator.train_on_batch(X_fake_vs_real, y1)

        #####################################
        ## TRAINING THE GENERATOR     ######
        ###################################

        # Create some noise
        noise = tf.random.normal(shape=[batch_size, latent_dim])

        # We want discriminator to belive that fake images are real
        y2 = tf.constant([[1.]] * batch_size)

        # Avois a warning
        discriminator.trainable = False

        GAN.train_on_batch(noise, y2)

    filename2 = 'model_%06d.h5' % (epoch + 1)
    generator.save(filename2)
    print('>Saved: %s ' % (filename2))

print("TRAINING COMPLETE")



noise = tf.random.normal(shape=[10, latent_dim])
plt.imshow(noise)
plt.show()

image = generator(noise)
plt.imshow(image[5])
plt.show()
