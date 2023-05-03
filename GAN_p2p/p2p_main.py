import os
import time
import numpy as np
import tensorflow as tf

from functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf, IC_normalization
from functions.p2p_discriminator_architecture import define_discriminator_512x32
from functions.p2p_generator_architecture import define_generator
from functions.p2p_gan_architecture import define_gan
from functions.p2p_train_architecture import train

########################################################################################################################
#   PATHS
########################################################################################################################
# For DelftBlue un-comment this...
path_data = r'/scratch/fcamposmontero/databases/512x32_20k/train'
path_results = r'/scratch/fcamposmontero/results_p2p/512x32_e200_s16k'

# For local run un-comment this...
#path_data = 'C:\\inpt\\synthetic_data\\512x32\\train'
#path_results = r'C:\inpt\GAN_p2p\results\test'

results_dir_path = os.path.join(path_results, 'results_summary.txt')

# Check the time and start the timers
time_current = time.strftime("%d/%m/%Y %H:%M:%S")

########################################################################################################################
#   CHOOSE THE  GEOMETRY, EPOCHS AND MISSING RATE
########################################################################################################################

# Resizing images, if needed
SIZE_X = 512
SIZE_Y = 32
no_rows = SIZE_Y
no_cols = SIZE_X

#miss_rate = 0.9868
#min_distance = 51
miss_rate = 0.99
min_distance = 51

# Number of epochs
n_epochs = 200


########################################################################################################################
#   PROCESS THE DATA AND DEFINE THE MODELS
########################################################################################################################

# Create empty containers for the target and source images
tar_images = []
src_images = []

# Read all the CSV files (cross-sections)
all_csv = read_all_csv_files(path_data)
# Generate the CPT-like data from the cross-sections
missing_data, full_data= apply_miss_rate_per_rf(all_csv, miss_rate, min_distance)
no_samples = len(all_csv)

# Reshape into matrix of the appropriate size for the cross-section
missing_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in missing_data])
full_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in full_data])
tar_images = np.reshape(full_data, (no_samples, no_rows, no_cols, 1))
src_images = np.reshape(missing_data, (no_samples, no_rows, no_cols, 1))

# Define input shape based on the loaded dataset
image_shape = src_images.shape[1:]

# Define the models
d_model = define_discriminator_512x32(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)

# Format the data to use in the models
data = [src_images, tar_images]

# Normalize the data to [-1 to 1]
dataset = IC_normalization(data)


########################################################################################################################
#   TRAIN THE GAN
########################################################################################################################
time_start = time.time()    # Start the timer

# Call the train function
train(path_results, d_model, g_model, gan_model, dataset, n_epochs, n_batch=1)

time_end = time.time()      # End the timer
execution_time = abs(time_start - time_end) # Calculate the run time


########################################################################################################################
#   GENERATE THE SUMMARY OF THE MODEL
########################################################################################################################
# Format time taken to run into> Hours : Minutes : Seconds
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = int(execution_time % 60)
time_str = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

# Save run details to file
with open(results_dir_path, "a") as file:
    file.write("Executed on: {}\n".format(time_current))
    file.write("Execution time: {}\n\n".format(time_str))
    file.write("The shape of a single input image is: {}\n".format(image_shape))
    file.write("The total number of training images is: {}\n".format(src_images.shape[0]))
    file.write("Number of epochs is: {}\n\n".format(n_epochs))
    file.write("Missing rate: {}\n\n".format(miss_rate))
    file.write("Minimum distance: {}\n\n".format(min_distance))

# Check if GPUs are available and write it to the summary
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    num_gpus = len(gpus)
    with open(results_dir_path, "a") as file:
        file.write("Running TensorFlow with {} GPU(s)\n".format(num_gpus))
else:
    with open(results_dir_path, "a") as file:
        file.write("Running TensorFlow on CPU\n")

# Save GAN model summaries to file
with open(results_dir_path, "a") as file:
    file.write(" \n")
    file.write("Generator summary\n")
    g_model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write(" \n\n")
    file.write("Discriminator summary\n")
    d_model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write(" \n")

