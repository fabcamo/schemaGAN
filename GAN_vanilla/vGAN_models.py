import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import randint, randn

from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.layers import InputLayer, Reshape, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#results_dir_path = r'C:\inpt\GAN_vanilla\results\test'
results_dir_path = r'/scratch/fcamposmontero/vGAN_res_sig'



# Discriminator architecture model
def discriminator_model(input_shape):
    model = Sequential() # Initiate sequential model
    model.add(InputLayer(input_shape=input_shape)) # Layer to be used as an entry point into a Network
    model.add(Flatten()) # Flatten the 2D input into 1D
    # Add the Dense layers> D512-D256-D128
    model.add(Dense(units=512, activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Dense(units=256, activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Dense(units=128, activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    # Add the FINAL dense layer to classify real or fake
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return model


# Generator architecture model
def generator_model(latent_dim):
    # Define how many units we need beforehand in the last layer before reshaping
    n_nodes = 28 * 28  # because the dimension of the output pictures should be 28x28

    # Build the sequential model
    model = Sequential() # Initiate sequential model
    model.add(InputLayer(input_shape=latent_dim)) # Layer to be used as an entry point into a Network
    # Add the Dense layers> D128-D256-D512
    model.add(Dense(units=128, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(units=256, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(units=512, activation=LeakyReLU(alpha=0.2)))
    # Final dense layer with units = n_nodes to be able to reshape as 28x28
    model.add(Dense(units=n_nodes, activation='sigmoid'))

    # Reshape from 1D to 2D to apply Conv2D
    model.add(Reshape((28, 28, 1)))
    # NOTE> GENERATOR MODEL IS NOT COMPILED HERE
    # Unlike the Discriminator, the Generator is trained in the GAN model
    return model


# Define the combined generator and discriminator model, for updating the generator
# Discriminator is trained separately so here only generator will be trained by keeping
# the discriminator constant.
def GAN_model(generator, discriminator):
    discriminator.trainable = False  # Set Discriminator as NOT trainable in the GAN
    # Connect the Generator and Discriminator
    model = Sequential() # Initiate the sequential model
    model.add(generator) # Add the Generator
    model.add(discriminator) # Add the Discriminator

    # Compile the model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# load MNIST numbers training images
def load_data():
    # Pull the train and test images and labels
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    all_images = np.concatenate([x_train, x_test])  # Concatenate all images into one variable
    all_labels = np.concatenate([y_train, y_test])  # Concatenate all labels into one variable
    return all_images, all_labels


# Normalize the data between -0 & 1
def normalize_data():
    all_data = load_data() # Grab all the data from calling the load_data function
    only_images = all_data[0].astype("float32") # Transform as type> float32
    norm_image = (only_images / 255) # Normalize between 0 and 1 for sigmiod activation
    return norm_image


# Call real samples and assign real labels to them (1)
def generate_real_samples(dataset, how_many):
    # Gather 'n' images from a range of 0 to the dataset size (7000)
    ix = randint(0, dataset.shape[0], how_many)
    # Assign those random images from the dataset to a container
    random_real_images = dataset[ix]

    # Generate class labels and assign them as real (1> real)
    labels = np.ones((how_many, 1))
    return random_real_images, labels


# Generate 'n' samples of noise as input for the generator
def generate_noise_vectors(latent_dim, how_many):
    # Generate points in the latent space
    noise = randn(latent_dim * how_many) # Returns 'n' number of random numbers from a std normal dist
    noise = noise.reshape(how_many, latent_dim) # Reshape into 'n' number of entries with size 'latent_dim'
    return noise


# use the generator to generate 'n' fake examples, with class labels
def generate_fake_samples(generator, latent_dim, how_many):
    noise = generate_noise_vectors(latent_dim, how_many) # Generate 'n' noise vectors
    fake_images = generator.predict(noise) # Create the fake images using the generator
    labels = np.zeros((how_many, 1)) # Create the labels as> 0 as these samples are fake.
    return fake_images, labels


# Pull the images and models for every 'X' amount of epochs
def summarize_performance(step, g_model, latent_dim, n_samples=25):
    # Generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, latent_dim, n_samples)

    # Scale all pixels from [0, 1] to [0, 255]
    X_fakeB = X_fakeB * 255

    # Plot images from the training dataset
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_fakeB[i], cmap='gray')
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    # Save plot to file
    plot_filename = os.path.join(results_dir_path, 'plot_{:06d}.png'.format(step + 1))
    plt.savefig(plot_filename)
    plt.close()

    # Save the generator model
    model_filename = os.path.join(results_dir_path, 'model_{:06d}.h5'.format(step + 1))
    g_model.save(model_filename)

    print('>Saved: {} and {}'.format(plot_filename, model_filename))



# Train the Generator and Discriminator
# We loop through a number of epochs to train our Discriminator by first selecting
# a random batch of images from our true/real dataset
# Then, generating a set of images using the generator
# Feed both set of images into the Discriminator
# Finally, set the loss parameters for both the real and fake images, as well as the combined loss
def train(generator, discriminator, gan, dataset, latent_dim, n_epochs, n_batch):
    # Define the batches for the training
    batch_per_epoch = int(dataset.shape[0] / n_batch) # How many batches per epoch [7000/128]
    half_batch = int(n_batch / 2) # Define the half batch size
    d_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list()
    d_epoch_hist, g_epoch_hist, a1_epoch_hist, a2_epoch_hist = list(), list(), list(), list()
    iterations = n_epochs * batch_per_epoch

    # The Discriminator model is updated for a half batch of real samples
    # and a half batch of fake samples, combined into a single batch

    # Manually enumerate epochs and batches
    for i in range(n_epochs):
        g_loss_all, d_loss_all, acc_real_all, acc_fake_all = 0.0, 0.0, 0.0, 0.0

        # Enumerate batches over the training set
        for j in range(batch_per_epoch):
            # TRAIN THE DISCRIMINATOR: on real and fake images, separately (half batch each)
            # Get randomly selected 'real' samples
            real_image, real_label = generate_real_samples(dataset, half_batch)
            # train_on_batch allows you to update weights based on a collection of samples you provide
            d_loss_real, d_acc_real = discriminator.train_on_batch(real_image, real_label)
            # Generate the fake images
            fake_image, fake_label = generate_fake_samples(generator, latent_dim, half_batch)
            # Update discriminator model weights
            d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_image, fake_label)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # TRAIN THE GENERATOR:
            # Prepare points in latent space as input for the generator
            noise = generate_noise_vectors(latent_dim, n_batch)
            # The generator wants the discriminator to label the generated samples as valid (ones)
            noise_fake_label = np.ones((n_batch, 1))

            # Generator is part of combined model where it got directly linked with the discriminator
            # Update the generator via the discriminator's error
            g_loss = gan.train_on_batch(noise, noise_fake_label)

            # Print losses on this batch
            print('Epoch>%d, Batch %d/%d, d=%.3f, g=%.3f' %
                  (i+1, j+1, batch_per_epoch, d_loss, g_loss))


            d_hist.append(d_loss)
            g_hist.append(g_loss)
            a1_hist.append(d_acc_real)
            a2_hist.append(d_acc_fake)

            g_loss_all += g_loss
            d_loss_all += d_loss
            acc_real_all += d_acc_real
            acc_fake_all += d_acc_fake

        epoch_loss_g = g_loss_all / j  # total generator loss for the epoch
        epoch_loss_d = d_loss_all / j  # total discriminator loss for the epoch
        epoch_acc_real = acc_real_all / j
        epoch_acc_fake = acc_fake_all / j
        g_epoch_hist.append(epoch_loss_g)
        d_epoch_hist.append(epoch_loss_d)
        a1_epoch_hist.append(epoch_acc_real)
        a2_epoch_hist.append(epoch_acc_fake)

        # Summarize model performance
        summarize_every_n_epochs = 1
        if i % summarize_every_n_epochs == 0:
            summarize_performance(i, generator, latent_dim, n_samples=25)
            plot_history(d_hist, g_hist, g_epoch_hist, d_epoch_hist,
                         a1_hist, a2_hist, a1_epoch_hist, a2_epoch_hist, i, n_epochs, iterations)




    # Save the generator model
    final_generator_path = os.path.join(results_dir_path, 'mnist_final_generator.h5')
    generator.save(final_generator_path)
    plot_history(d_hist, g_hist, g_epoch_hist, d_epoch_hist,
                         a1_hist, a2_hist, a1_epoch_hist, a2_epoch_hist, i, n_epochs, iterations)
    # Save results to dataframe and CSV file
    df = pd.DataFrame({'disc_loss': d_hist, 'gen_loss': g_hist, 'acc_real': a1_hist, 'acc_fake': a2_hist})
    csv_file = os.path.join(results_dir_path, 'results_loss.csv')
    df.to_csv(csv_file, index=False)


# create a line plot of loss for the gan and save to file
def plot_history(d_hist, g_hist, g_epoch_hist, d_epoch_hist, a1_hist, a2_hist, a1_epoch_hist, a2_epoch_hist, step, n_epochs, iterations):
    # create figure for loss
    plt.figure(figsize=(10, 4))
    plt.plot(d_hist, label='disc', color='black')
    plt.plot(g_hist, label='gen', color='darkgray')
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # set x-axis limits
    plt.xlim([0, iterations])
    # Save plot to file
    plot_loss = os.path.join(results_dir_path, 'plot_loss_{:06d}.png'.format(step + 1))
    plt.savefig(plot_loss)
    plt.close()

    # create figure for loss per epoch
    plt.figure(figsize=(10, 4))
    plt.plot(d_epoch_hist, label='disc', color='black')
    plt.plot(g_epoch_hist, label='gen', color='darkgray')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # set x-axis limits
    plt.xlim([0, n_epochs])
    # Save plot to file
    plot_loss = os.path.join(results_dir_path, 'plot_loss_epoch_{:06d}.png'.format(step + 1))
    plt.savefig(plot_loss)
    plt.close()

    # create figure for accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(a1_hist, label='acc-real', color='black', alpha=0.8)
    plt.plot(a2_hist, label='acc-fake', color='darkgray', alpha=0.8)
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    # set x-axis limits
    plt.xlim([0, iterations])
    # set y-axis limits
    plt.ylim([0, 1])
    # Save plot to file
    plot_acc = os.path.join(results_dir_path, 'plot_acc_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()

    # create figure for accuracy per epoch
    plt.figure(figsize=(10, 4))
    plt.plot(a1_epoch_hist, label='acc-real', color='black', alpha=0.8)
    plt.plot(a2_epoch_hist, label='acc-fake', color='darkgray', alpha=0.8)
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # set x-axis limits
    plt.xlim([0, n_epochs])
    # set y-axis limits
    plt.ylim([0, 1])
    # Save plot to file
    plot_acc = os.path.join(results_dir_path, 'plot_acc_epoch_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()

    # create figure with two subplots for accuracy
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    # plot data on each subplot
    ax1.plot(a1_hist, label='Real p/iter', color='black')
    ax1.plot(a2_hist, label='Fake p/iter', color='dimgray')
    ax2.plot(a1_epoch_hist, label='Real p/epoch', color='gray')
    ax2.plot(a2_epoch_hist, label='Fake p/epoch', color='silver')
    # set labels and legends for each subplot
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    # save plot to file
    plot_acc = os.path.join(results_dir_path, 'plot_acc_combined_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()

    # create figure with two subplots for accuracy
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    # plot data on each subplot
    ax1.plot(d_hist, label='D p/iter', color='black')
    ax1.plot(g_hist, label='G p/iter', color='dimgray')
    ax2.plot(d_epoch_hist, label='D p/epoch', color='gray')
    ax2.plot(g_epoch_hist, label='G p/epoch', color='silver')
    # set labels and legends for each subplot
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Loss')
    ax1.set_ylim([0, 4])
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    # save plot to file
    plot_acc = os.path.join(results_dir_path, 'plot_loss_combined_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()