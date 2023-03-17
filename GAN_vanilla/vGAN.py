import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import randint, randn

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.layers import InputLayer, Reshape, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Check if GPUs are available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Import data from MNIST numbers dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
all_images = np.concatenate([x_train, x_test])      # concatenate all images into one variable
all_labels = np.concatenate([y_train, y_test])      # concatenate all labels into one variable


# Plot images from the training dataset
for i in range(25):
    plt.subplot(5, 5, 1 + i)        # define subplot
    plt.axis('off')                 # turn off axis
    plt.imshow(all_images[i], cmap='Greys')
plt.show()                          # to show the image
plt.clf()                           # To clear the image memory


##### HYPER-PARAMETERS ##########################################################################################
no_channels = 1         # number of channels in the image
latent_dim = 128        # user defined number as input to the generator
batch_size = 32         # user defined batch size
n_samples = 1
n_epochs = 100
n_batch = 128

# Check the image size
print('The shape of a single image is: ',all_images[0].shape)
input_shape = all_images[0].shape

##################################################################################################################

# Discriminator architecture model
def discriminator_model(input_shape):
    model = Sequential()
    # Layer to be used as an entry point into a Network
    model.add(InputLayer(input_shape=input_shape))
    # Flatten the 2D input
    model.add(Flatten())
    # Add the Dense layers
    model.add(Dense(units=512, activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Dense(units=256, activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Dense(units=128, activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    # Add FINAL fake or real layer
    model.add(Dense(units=1, activation='sigmoid'))

    # Define the optimizer
    opt = Adam(lr=0.0002, beta_1=0.5)
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# Generator architecture model
def generator_model(latent_dim):
    # Define how many units we need beforehand in the last layer before reshaping
    n_nodes = 28 * 28  # because the dimension of the output pictures should be 28x28

    # Build the sequential model
    model = Sequential()
    # Layer to be used as an entry point into a Network
    model.add(InputLayer(input_shape=latent_dim))
    # Add Dense layer starting from a given number of neurons
    model.add(Dense(units=128, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(units=256, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(units=512, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(units=n_nodes, activation=LeakyReLU(alpha=0.2)))

    # Reshape from 1D to 2D to apply Conv2D
    model.add(Reshape((28, 28, 1)))
    #NOTE> MODEL NOT COMPILED
    return model  # Model not compiled as it is not directly trained like the discriminator.
    # Generator is trained via GAN combined model.



# Define the combined generator and discriminator model, for updating the generator
# Discriminator is trained separately so here only generator will be trained by keeping
# the discriminator constant.
def GAN_model(generator, discriminator):
    discriminator.trainable = False  # Discriminator is trained separately. So set to not trainable.
    # connect generator and discriminator
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model



# load MNIST numbers training images
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    all_images = np.concatenate([x_train, x_test])  # concatenate all images into one variable
    all_labels = np.concatenate([y_train, y_test])  # concatenate all labels into one variable

    return all_images, all_labels



# Normalize the data between -1 & 1
def normalize_data():
    all_data = load_data()      # grab all the data from calling the load_data function
    only_images = all_data[0].astype("float32")     # transform as type> float32
    norm_image = (only_images / 127.5) - 1          # normalize between -1 and 1
    return norm_image


# pick a batch of random real samples to train the GAN
# In fact, we will train the GAN on a half batch of real images and another
# half batch of fake images.
# For each real image we assign a label 1 and for fake we assign label 0.
def generate_real_samples(dataset, no_samples):
    # gather no_samples of images from a range of 0 to the dataset size (7000)
    ix = randint(0, dataset.shape[0], no_samples)
    # assign those random images from the dataset to a container
    random_images = dataset[ix]

    # generate class labels and assign
    labels = np.ones((no_samples, 1))  # label=1 indicating they are real
    return random_images, labels


# generate n_samples number of latent vectors as input for the generator
def generate_noise_vectors(latent_dim, no_samples):
    # generate points in the latent space
    noise = randn(latent_dim * no_samples) # returns n number of random numbers from a std normal dist
    # reshape into a batch of inputs for the network
    noise = noise.reshape(no_samples, latent_dim) # reshape into n number of entries with size latent_dim
    return noise


# use the generator to generate n fake examples, with class labels
# call the generator with> latent_dim and number of samples as input.
# Use the above latent point generator to generate latent points.
def generate_fake_samples(generator, latent_dim, no_samples):
    # # generate n noise vectors
    noise = generate_noise_vectors(latent_dim, no_samples)
    # create the fake images using the generator
    fake_images = generator.predict(noise)
    # create the labels as> 0 as these samples are fake.
    labels = np.zeros((no_samples, 1))  # Label=0 indicating they are fake
    return fake_images, labels



def summarize_performance(step, g_model, latent_dim, n_samples=25):
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale all pixels from [-1,1] to [0,1]
    X_fakeB = (X_fakeB + 1) / 2.0

    # Plot images from the training dataset
    for i in range(25):
        plt.subplot(5, 5, 1 + i)  # define subplot
        plt.axis('off')  # turn off axis
        plt.imshow(X_fakeB[i], cmap='Greys')
    plt.show()  # to show the image
    plt.clf()  # To clear the image memory

    # save plot to file
    filename1 = '/results/plot_%06d.png' % (step + 1)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
    filename2 = '/results/model_%06d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))





# train the generator and discriminator
# We loop through a number of epochs to train our Discriminator by first selecting
# a random batch of images from our true/real dataset.
# Then, generating a set of images using the generator.
# Feed both set of images into the Discriminator.
# Finally, set the loss parameters for both the real and fake images, as well as the combined loss.
def train(generator, discriminator, gan, dataset, latent_dim, n_epochs, n_batch):

    # Define the batches for the training
    batch_per_epoch = int(dataset.shape[0] / n_batch) # how many batches per epoch [7000/128]
    half_batch = int(n_batch / 2) # define the half batch size

    # the discriminator model is updated for a half batch of real samples
    # and a half batch of fake samples, combined a single batch

    # manually enumerate epochs and batches
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(batch_per_epoch):
            # TRAIN THE DISCRIMINATOR: on real and fake images, separately (half batch each)
            # get randomly selected 'real' samples
            real_image, real_label = generate_real_samples(dataset, half_batch)
            # train_on_batch allows you to update weights based on a collection of samples you provide
            d_loss_real, _ = discriminator.train_on_batch(real_image, real_label)
            # generate the fake images
            fake_image, fake_label = generate_fake_samples(generator, latent_dim, half_batch)
            # update discriminator model weights
            d_loss_fake, _ = discriminator.train_on_batch(fake_image, fake_label)

            # TRAIN THE GENERATOR:
            # prepare points in latent space as input for the generator
            noise = generate_noise_vectors(latent_dim, n_batch)
            # The generator wants the discriminator to label the generated samples as valid (ones)
            noise_fake_label = np.ones((n_batch, 1))

            # Generator is part of combined model where it got directly linked with the discriminator
            # update the generator via the discriminator's error
            g_loss = gan.train_on_batch(noise, noise_fake_label)

            # Print losses on this batch
            print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, batch_per_epoch, d_loss_real, d_loss_fake, g_loss))

            # summarize model performance
            if (i + 1) % (batch_per_epoch * 10) == 0:
                summarize_performance(i, generator, dataset)
    # save the generator model
    generator.save('/results/mnist_final generator.h5')


###################################################################
# Train the GAN

# size of the latent space
latent_dim = 100
# create the discriminator
discriminator = discriminator_model(input_shape)
# create the generator
generator = generator_model(latent_dim)
# create the gan
gan_model = GAN_model(generator, discriminator)
# load image data
dataset = normalize_data()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs, n_batch)

################################################################################

# Now, let us load the generator model and generate images

from keras.models import load_model
from numpy.random import randn


# Plot generated images
def show_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
    plt.show()


# load model
model = load_model('cifar_generator_250epochs.h5')  # Model trained for 100 epochs
# generate images
latent_points = generate_noise_vectors(100, 25)  # Latent dim and n_samples
# generate images
X = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0

import numpy as np

X = (X * 255).astype(np.uint8)

# plot the result
show_plot(X, 5)

# Note: CIFAR10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse,
# ship, truck
