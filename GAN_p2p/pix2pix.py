import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
import datetime

##### READING THE DATA FUNCTIONS #####################################################

# read all csv data
def read_all_csv_files(directory):
    directory = os.path.join(directory)    # Join the directory path with the os path separator
    dfs = []    # Create an empty list to store the read dataframes

    # Iterate through all the files in the directory using os.walk()
    for root, dirs, files in os.walk(directory):
        for file in files:  # Iterate through all the files and directories in the current root directory
            if file.endswith(".csv"):   # Check if the file ends with .csv
                df = pd.read_csv(os.path.join(directory, file), delimiter=',')
                dfs.append(df)  # Append the dataframe to the list of dataframes

    return dfs      # Return the list of dataframes


def apply_miss_rate_per_rf(dfs, miss_rate=0.8):
    missing_data, full_data = [], []     # Create two empty lists to store missing and full data
    value_name = 'IC'   # Set value_name to 'IC'

    # Iterate through each random forest in the list
    for counter, rf in enumerate(dfs):
        data_z = []     # Create an empty list to store data for each value of x
        grouped = rf.groupby("z")   # Group the rows of the random field by the value of x

        # Iterate through each group
        for name, group in grouped:
            data_z.append(list(group[value_name]))  # Append the 'IC' column of the group to the data_x list

        data_z = np.array(data_z, dtype=float)  # Convert the data_x list to a numpy array of type float
        no, dim = data_z.shape  # Get the number of rows and columns in the data_x array
        data_m = remove_random_columns(data_z, miss_rate)   # Call the remove_random_columns function to remove columns from data_x
        missing_data.append(data_m) # Append the missing data to the missing_data list
        full_data.append(data_z)    # Append the full data to the full_data list

    # Return the missing_data and full_data lists
    return missing_data, full_data


def load_and_normalize_RFs_in_folder(directory):
    dfs = read_all_csv_files(directory)
    train_input, train_output = apply_miss_rate_per_rf(dfs)
    # Careful, here is 64 rows and 256 columns
    train_input = np.array([np.reshape(i, (64, 256)).astype(np.float32) for i in train_input])
    train_output = np.array([np.reshape(i, (64, 256)).astype(np.float32) for i in train_output])
    maximum_value = max_IC_value
    train_output = np.array(train_output) / maximum_value
    train_input = np.array(train_input) / maximum_value
    return np.array([train_input, train_output]).astype(np.float32)
    # train_input > the RFs with 80% missing (the input to generate later with AI)
    # train_output > the complete RFs


def remove_random_columns(data_z, miss_rate):
    dim_choice = int(data_z.shape[0])
    missing_columns_index = random.sample(range(dim_choice), int(miss_rate*dim_choice))
    data_m = np.ones_like(data_z)
    for column_index in missing_columns_index:
        data_m[column_index, :] = np.zeros_like(data_m[column_index, :])
    miss_list = np.multiply(data_z, data_m)
    return miss_list

##### GENERATOR AND DISCRIMINATOR ########################################################

# downsample from pix2pix
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


# upsample from pix2pix
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


# Generator from pix2pix
def Generator():
  inputs = tf.keras.layers.Input(shape=[64, 256, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# generator loss from pix2pix
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


# Discriminator from pix2pix
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[64, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[64, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


# discriminator loss from pix2pix
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


###########################################################################################

# train step from pix2pix
@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)


# fit from pix2pix
def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)


def set_up_and_train_2d_model():
    all_data = read_all_csv_files("inpt\\synthetic_data\\cs2d\\train")

    dataset = load_and_normalize_RFs_in_folder("inpt\\synthetic_data\\cs2d\\train")
    input_dataset = tf.convert_to_tensor(tf.constant(dataset[0]))
    train_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset)
    train_input_dataset = train_input_dataset.batch(BATCH_SIZE)
    target_dataset = tf.convert_to_tensor(tf.constant(dataset[1]))
    train_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset)
    train_target_dataset = train_target_dataset.batch(BATCH_SIZE)
    train_dataset = tf.data.Dataset.zip((train_input_dataset, train_target_dataset))

    test_dataset = load_and_normalize_RFs_in_folder("inpt\\synthetic_data\\cs2d\\test")
    input_dataset_test = tf.convert_to_tensor(tf.constant(test_dataset[0]))
    test_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset_test)
    test_input_dataset = test_input_dataset.batch(BATCH_SIZE)
    target_dataset_test = tf.convert_to_tensor(tf.constant(test_dataset[1]))
    test_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset_test)
    test_target_dataset = test_target_dataset.batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.zip((test_input_dataset, test_target_dataset))

    #validation_dataset = load_validation_data("data\cond_rf\\validation_final")
    #input_dataset_validation = tf.convert_to_tensor(tf.constant(validation_dataset[0]))
    #validation_input_dataset = tf.data.Dataset.from_tensor_slices(input_dataset_validation)
    #validation_input_dataset = validation_input_dataset.batch(BATCH_SIZE)
    #target_dataset_validation = tf.convert_to_tensor(tf.constant(validation_dataset[1]))
    #validation_target_dataset = tf.data.Dataset.from_tensor_slices(target_dataset_validation)
    #validation_target_dataset = validation_target_dataset.batch(BATCH_SIZE)
    #validation_dataset = tf.data.Dataset.zip((validation_input_dataset, validation_target_dataset))

    fit(train_dataset, test_dataset,None,  steps=12000)

    print(1)
    # TODO plot against all the training inputs
    all_images(generator, test_dataset)
    # TODO plot diff
    dataset = [value for counter, value in enumerate(test_dataset)]
    predictions = [generator(data[0], training=True) for data in dataset]
    data_diff = [data[1].numpy()[0,:,:] - predictions[counter].numpy()[0,:,:,0] for counter, data in enumerate(dataset)]

    mean_axis_error = np.mean(np.array(np.abs(data_diff)), axis=0)
    plt.clf()
    plt.imshow(mean_axis_error)
    plt.colorbar
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(mean_axis_error.T)
    ax.set_xlabel("Mean error per pixel at the end of training")
    fig.colorbar(im, orientation="horizontal")
    plt.show()

    # TODO calculate all the diffs
    mean_error = np.mean(np.abs(np.array(data_diff)).flatten())
    std_error = np.std(np.array(data_diff).flatten())
    print(f"stats mean abserror {mean_error} std : {std_error}")


###########################################################################################


path = '/inpt/synthetic_data/cs2d/train/cs_0.png'
directory = '/inpt/synthetic_data/cs2d/train'


data = read_all_csv_files(directory)
merged_data = pd.concat(data)
max_IC_value = merged_data["IC"].max()
print("The maximum value of the 'IC' column in the list of dataframes is:", max_IC_value)



missing_data, full_data = apply_miss_rate_per_rf(data)

data_set = load_and_normalize_RFs_in_folder(directory)

print('train input normalized')
print(data_set[1][0].shape)
test = data_set[1][0]
print(test)
# Swap the x-axis and y-axis using transpose()

plt.imshow(test)
plt.colorbar()
plt.show()


##################################################################################################
print('##################################################')

def load(path):
    path = os.path.join(path)
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    input_image = image

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)

    return input_image


inp = load(path)
# Casting to int for matplotlib to display the images
plt.figure()
plt.imshow(inp)


#inp = full_data[0]

# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1

# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 64

# Output channels > IC = 1
OUTPUT_CHANNELS = 1
LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# define optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)



log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


generator = Generator()
discriminator = Discriminator()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

set_up_and_train_2d_model()

# # test the downsample
# down_model = downsample(3, 4)
# down_result = down_model(tf.expand_dims(inp, 0))
# print (down_result.shape)
#
# # test the upsample
# up_model = upsample(3, 4)
# up_result = up_model(down_result)
# print (up_result.shape)