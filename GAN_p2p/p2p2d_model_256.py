import os
import pandas as pd
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot as plt

########################################################################################################################
# Define generator, discriminator, gan and other helper functions
# We will use functional way of defining model and not sequential
# as we have multiple inputs; both images and corresponding labels.
########################################################################################################################

# Since pix2pix is a conditional GAN, it takes 2 inputs - image and corresponding label
# For pix2pix the label will be another image.

# define the standalone discriminator model
# Given an input image, the Discriminator outputs the likelihood of the image being real.
# Binary classification - true or false (1 or 0). So using sigmoid activation.
# Think of discriminator as a binary classifier that is classifying images as real/fake.

# From the paper C64-C128-C256-C512
# After the last layer, conv to 1-dimensional output, followed by a Sigmoid function.
########################################################################################################################


# read all csv data to train on
def read_all_csv_files(directory):
    directory = os.path.join(directory)    # Join the directory path with the os path separator
    csv_data = []    # Create an empty list to store the read dataframes

    # Iterate through all the files in the directory using os.walk()
    for root, dirs, files in os.walk(directory):
        for file in files:  # Iterate through all the files and directories in the current root directory
            if file.endswith(".csv"):   # Check if the file ends with .csv
                df = pd.read_csv(os.path.join(directory, file), delimiter=',')
                csv_data.append(df)  # Append the dataframe to the list of dataframes

    return csv_data      # Return the list of dataframes


# Create the conditional input CPT-like data
def apply_miss_rate_per_rf(dfs, miss_rate, min_distance):
    missing_data, full_data = [], []     # Create two empty lists to store missing and full data
    value_name = 'IC'   # Set value_name to 'IC'

    # Iterate through each random field in the list
    for counter, rf in enumerate(dfs):
        data_z = []     # Create an empty list to store data for each value of x
        grouped = rf.groupby("z")   # Group the rows of the random field by the value of x

        # Iterate through each group
        for name, group in grouped:
            data_z.append(list(group[value_name]))  # Append the 'IC' column of the group to the data_x list

        data_z = np.array(data_z, dtype=float)  # Convert the data_x list to a numpy array of type float
        data_m = remove_random_columns(data_z, miss_rate, min_distance)   # Call the remove_random_columns function to remove columns from data_x
        missing_data.append(data_m) # Append the missing data to the missing_data list
        full_data.append(data_z)    # Append the full data to the full_data list

    # Return the missing_data and full_data lists
    return missing_data, full_data




# Remove at random a user defined percentage of columns from the matrix
def remove_random_columns(data_z, miss_rate, min_distance):
    # Transpose the input data to operate on columns instead of rows
    data_z = np.transpose(data_z)
    # Create a matrix of ones that will be used to indicate missing data
    data_m = np.zeros_like(data_z)

    # Returns which columns to keep from miss_rate and min_distance
    columns_to_keep_index = check_min_spacing(data_z, miss_rate, min_distance)

    # Set the values in data_m to 0 for the columns that were selected for removal
    for column_index in columns_to_keep_index:
        data_m[column_index, :] = np.ones_like(data_m[column_index, :])

    # Remove a random number of rows from the bottom from each column
    data_m = remove_random_depths(data_z, data_m)

    # Multiply the original data by the missing data indicator to create the final output
    miss_list = np.multiply(data_z, data_m)
    # Transpose the output back to its original orientation
    miss_list = np.transpose(miss_list)

    return miss_list


# Select the columns to keep for each cross-section
def check_min_spacing(data_z, miss_rate, min_distance):
    # Choose the dimension based on the number of columns in the transposed data
    all_columns = int(data_z.shape[0])  # [256]
    # Calculate how many columns (indexes) we need according to the missing rate
    no_missing_columns = int(miss_rate * all_columns)  # number of missing columns
    no_columns_to_keep = abs(no_missing_columns - all_columns)  # number of columns to keep [like CPTs]

    columns_to_keep_index = []  # Empty container for the missing indexes

    # Loop until the columns_to_keep_index list is full according to the selected percentage
    while len(columns_to_keep_index) != no_columns_to_keep:
        # Generate a random column index from an uniform distribution
        rand_index = int(np.random.uniform(0, all_columns))
        # Define the range of indexes to check for duplicates according to the min_distance defined
        range_to_check = range(rand_index - min_distance, rand_index + min_distance + 1)

        if rand_index in columns_to_keep_index:  # Check if the rand_index is already in columns_to_keep_index
            pass  # if it is> do nothing, restart the while-loop
        else:
            # Check if none of the indexes in the range are already in columns_to_keep_index list
            if all(index not in columns_to_keep_index for index in range_to_check):
                columns_to_keep_index.append(rand_index)  # if true, append the rand_index to the list
            else:
                print('No space to accommodate random index, RETRYING')

    return columns_to_keep_index


# Remove a random amount of data from the bottom of each column in the matrix
def remove_random_depths(data_z, data_m):
    data_length = data_z.shape[0]  # grab the length of the cross-section [256 columns]
    data_depth = data_z.shape[1]   # grab the depth of the cross-section [64 rows]
    for j in range(data_length):   # iterate over columns
        # generate a random number with bias towards lower numbers
        # this will be the number of rows to transform to zero
        n_rows = int(np.random.triangular(0,0, data_depth/2))
        if n_rows > 0:
            # for every j column, select the last n_rows
            # replace those n_rows with zeros
            data_m[j, -n_rows:] = np.zeros(n_rows)

    return data_m





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


# disc_model = define_discriminator((256,256,3))
# plot_model(disc_model, to_file='disc_model.png', show_shapes=True)

##############################
# Now define the generator - in our case we will define a U-net
# define an encoder block to be used in generator
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


def define_encoder_block_mod(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block to be used in generator
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


def decoder_block_mod(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(1, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model - U-net
def define_generator(image_shape=(64, 256, 1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block_mod(e5, 512)
    e7 = define_encoder_block_mod(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model: CD512-CD512-CD512-C512-C256-C128-C64
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block_mod(d1, e6, 512)
    d3 = decoder_block_mod(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(image_shape[2], (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
        d7)  # Modified
    out_image = Activation('tanh')(g)  # Generates images in the range -1 to 1. So change inputs also to -1 to 1
    # define model
    model = Model(in_image, out_image)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False  # Descriminator layers set to untrainable in the combined GAN but
            # standalone descriminator will be trainable.

    # define the source image
    in_src = Input(shape=image_shape)
    # suppy the image as input to the generator
    gen_out = g_model(in_src)
    # supply the input image and generated image as inputs to the discriminator
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and disc. output as outputs
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)

    # IMPORTAT TO UNDERSTAND THISSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
    # Total loss is the weighted sum of adversarial loss (BCE) and L1 loss (MAE)
    # Authors suggested weighting BCE vs L1 as 1:100.
    model.compile(loss=['binary_crossentropy', 'mae'],
                  optimizer=opt, loss_weights=[1, 100])
    return model


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape*4, 1))
    return [X1, X2], y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape*4, 1))
    return X, y


def preprocess_data(data):
    # load compressed arrays
    # unpack arrays
    X1, X2 = data[0], data[1]
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]



# generate samples and save as a plot and save the model
# GAN models do not converge, we just want to find a good balance between
# the generator and the discriminator. Therefore, it makes sense to periodically
# save the generator model and check how good the generated image looks.
def summarize_performance(step, g_model, dataset, n_samples=3):
    print('... Saving a summary')
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    # plot real source images
    for i in range(n_samples):
        ax = plt.subplot(3, n_samples, 1 + i)
        # plt.axis('off')
        plt.imshow(X_realA[i])
        if i == 0:
            plt.ylabel('Real Source', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xticklabels(['0', '64', '128', '192', '256'], fontsize=6)
        ax.set_yticks([0, 32, 64])
        ax.set_yticklabels(['0', '32', '64'], fontsize=6)

    # plot generated target image
    for i in range(n_samples):
        ax = plt.subplot(3, n_samples, 1 + n_samples + i)
        # plt.axis('off')
        plt.imshow(X_fakeB[i])
        if i == 0:
            plt.ylabel('Generated Target', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xticklabels(['0', '64', '128', '192', '256'], fontsize=6)
        ax.set_yticks([0, 32, 64])
        ax.set_yticklabels(['0', '32', '64'], fontsize=6)

    # plot real target image
    for i in range(n_samples):
        ax = plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
        # plt.axis('off')
        plt.imshow(X_realB[i])
        if i == 0:
            plt.ylabel('Real Target', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xticklabels(['0', '64', '128', '192', '256'], fontsize=6)
        ax.set_yticks([0, 32, 64])
        ax.set_yticklabels(['0', '32', '64'], fontsize=6)

    filename1 = 'plot_%06d.png' % (step + 1)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step + 1)
    #g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs

    d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list(), list()

    # Manually enumerate epochs and batches
    for i in range(n_epochs):
        # Enumerate batches over the training set
        for j in range(bat_per_epo):
            # TRAIN THE DISCRIMINATOR
            # select a batch of real samples
            [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
            # update discriminator for real samples
            d_loss_real, d_acc_real = d_model.train_on_batch([X_realA, X_realB], y_real)
            # generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
            # update discriminator for generated samples
            d_loss_fake, d_acc_fake = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

            # TRAIN THE GENERATOR
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

            # summarize performance
            # Print losses on this batch
            print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))

            # Storing the losses and accuracy of the iterations.
            d1_hist.append(d_loss_real)
            d2_hist.append(d_loss_fake)
            d_hist = np.add(d1_hist, d2_hist).tolist()
            g_hist.append(g_loss)
            a1_hist.append(d_acc_real)
            a2_hist.append(d_acc_fake)

        # summarize model performance
        summarize_every_n_epochs = 5
        if i % summarize_every_n_epochs == 0:
            summarize_performance(i, g_model, dataset)
            plot_history(d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist)

    # Save the generator model
    final_generator_path = 'final_generator.h5'
    g_model.save(final_generator_path)
    plot_history(d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist)


'''

    # manually enumerate epochs
    for i in range(n_steps):
        # TRAIN THE DISCRIMINATOR
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # update discriminator for real samples
        d_loss_real, d_acc_real = d_model.train_on_batch([X_realA, X_realB], y_real)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for generated samples
        d_loss_fake, d_acc_fake = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        # TRAIN THE GENERATOR
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('Training>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss_real, d_loss_fake, g_loss))

        # Storing the losses and accuracy of the iterations.
        d1_hist.append(d_loss_real)
        d2_hist.append(d_loss_fake)
        d_hist = np.add(d1_hist, d2_hist).tolist()
        g_hist.append(g_loss)
        a1_hist.append(d_acc_real)
        a2_hist.append(d_acc_fake)


        # summarize model performance
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)
            plot_history(d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist)


'''

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='d-real')
    plt.plot(d2_hist, label='d-fake')
    plt.plot(d_hist, label='d-total')
    plt.plot(g_hist, label='gen')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plot discriminator accuracy
    plt.subplot(2, 1, 2)
    plt.plot(a1_hist, label='acc-real')
    plt.plot(a2_hist, label='acc-fake')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Save plot to file
    plot_losses = 'plot_losses.png'

    plt.savefig(plot_losses, bbox_inches='tight')
    plt.close()


# Plot the input, generated and original images
def plot_images(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img, np.abs(gen_img-tar_img)))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Input', 'Output-Generated', 'Original', 'error']
    ranges_vmin_vmax = [[1.6, 4], [1.6, 4], [1.6, 4], [0, 2]]
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        plt.subplot(1, 4, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i,:,:,0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])
        # show title
        plt.title(titles[i])
    plt.show()