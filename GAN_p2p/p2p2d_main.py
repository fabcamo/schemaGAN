import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from functions10.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf, preprocess_data
from functions10.p2p_discriminator_architecture import define_discriminator_512x32, define_discriminator
from functions10.p2p_generator_architecture import define_generator
from functions10.p2p_gan_architecture import define_gan
from functions10.p2p_train_architecture import train




# Resizing images, if needed
SIZE_X = 512
SIZE_Y = 32
no_rows = SIZE_Y
no_cols = SIZE_X

# Define the paths
path = r'/scratch/fcamposmontero/512x32_reduced'
path_results = r'/scratch/fcamposmontero/p2p_512x32_results10'

#path = r'/scratch/fcamposmontero/512x32/training'
#path_results = r'/scratch/fcamposmontero/p2p_512x32_results'

#path = 'C:\\inpt\\synthetic_data\\512x32'
#path_results = r'C:\inpt\GAN_p2p\results\test'

#results_dir_path = r'/scratch/fcamposmontero/p2p_512x32_results/results_summary.txt'
results_dir_path = os.path.join(path_results, 'results_summary.txt')

# Check the time and start the timers
time_current = time.strftime("%d/%m/%Y %H:%M:%S")



#miss_rate = 0.9868
#min_distance = 51
miss_rate = 0.90
min_distance = 6

# Number of epochs
n_epochs = 500

# Capture training image info as a list
tar_images = []

# Capture mask/label info as a list
src_images = []

all_csv = read_all_csv_files(path)
missing_data, full_data= apply_miss_rate_per_rf(all_csv, miss_rate, min_distance)
no_samples = len(all_csv)

missing_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in missing_data])
full_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in full_data])

tar_images = np.reshape(full_data, (no_samples, no_rows, no_cols, 1))
src_images = np.reshape(missing_data, (no_samples, no_rows, no_cols, 1))



# define input shape based on the loaded dataset
image_shape = src_images.shape[1:]




# define the models
d_model = define_discriminator_512x32(image_shape)
g_model = define_generator(image_shape)

# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

# Define data
# load and prepare training images
data = [src_images, tar_images]


# Preprocess data to change input range to values between -1 and 1
# This is because the generator uses tanh activation in the output layer
# And tanh ranges between -1 and 1
dataset = preprocess_data(data)

#################################################################################################################
#   TRAIN THE GAN
#################################################################################################################

time_start = time.time() # start the timer

train(d_model, g_model, gan_model, dataset, n_epochs, n_batch=1)
# Reports parameters for each batch (total 1600) for each epoch.
# For 10 epochs we should see 16000

time_end = time.time() # End the timer
# Execution time of the model
execution_time = abs(time_start - time_end) # Calculate the run time
print("Execution time is: ", execution_time)




#################################################################################################################
#   GENERATE THE SUMMARY OF THE MODEL
#################################################################################################################
# Format time taken to run into> Hours : Minutes : Seconds
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = int(execution_time % 60)
time_str = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

# Save run details to file
with open(results_dir_path, "a") as file:
    file.write("Executed on: {}\n".format(time_current))
    file.write("execution time: {}\n\n".format(time_str))
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








'''
# To print a sample of the input and the original
n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(src_images[i], cmap='viridis')
# plot target image
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(tar_images[i], cmap='viridis')
plt.show()

'''