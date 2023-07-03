import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from GAN_p2p.interpolation.interpolation_utils import get_cptlike_data, format_source_images
from GAN_p2p.interpolation.interpolation_utils import generate_gan_image, compute_errors_gan_only
from GAN_p2p.functions.p2p_process_data import load_remove_reshape_data, IC_normalization


# Check the time and start the timers
time_current = time.strftime("%d/%m/%Y %H:%M:%S")

########################################################################################################################
#   PATH FOR THE VALIDATION DATA AND MODEL TO EVALUATE
########################################################################################################################
# For DelftBlue
#path_validation = r'/scratch/fcamposmontero/databases/512x32_9010_10k/validation'
#path_to_model_to_evaluate = r'/scratch/fcamposmontero/results_p2p/512x32_e1000_s9000/99.51_9010'
#path_results = r'/scratch/fcamposmontero/results_p2p/512x32_e1000_s9000/99.51_9010/val_w5'
# For local
path_validation = 'C:\\inpt\\synthetic_data\\512x32\\validation'
path_to_model_to_evaluate = 'C:\\inpt\\GAN_p2p\\results\some_generators'
path_results = 'C:\\inpt\\GAN_p2p\\results\\test\\validation'


########################################################################################################################
#   GENERATE SEED
########################################################################################################################
# Generate a random seed using NumPy
seed = np.random.randint(20220412, 20230412)
# Set the seed for NumPy's random number generator
np.random.seed(seed)

########################################################################################################################
#   CHOOSE THE DIMENSIONS AND MISSING RATE
########################################################################################################################
# Images size
SIZE_X = 512
SIZE_Y = 32
no_rows = SIZE_Y
no_cols = SIZE_X
# Choose missing rate
miss_rate = 0.99
min_distance = 51


########################################################################################################################
#   LOAD THE VALIDATION DATA
########################################################################################################################
# load all images, convert to cptlike and reshape them
tar_images, src_images = load_remove_reshape_data(path_validation, miss_rate, min_distance, no_rows, no_cols)

# Grab the number of validation images
no_validation_images = src_images.shape[0]

# Create the array of source and target images
data = [src_images, tar_images]

# Normalize the data from [0 - 4.5] to [-1 to 1]
dataset = IC_normalization(data)
[input_img, orig_img] = dataset

# Get the coordinates and the pixel values for the cptlike data
coords_all, pixel_values_all = get_cptlike_data(src_images)

# Format the original image and the cptlike image for ploting
original_images, cptlike_img = format_source_images(dataset)

########################################################################################################################

# Use os.listdir to get a list of all the files in the directory
all_files = os.listdir(path_to_model_to_evaluate)
# Use list comprehension to filter out only the files that end with '.h5'
model_files = [file for file in all_files if file.endswith('.h5')]
# Use a lambda function to extract the XXXXXX part of the filename and sort the files by it
model_files = sorted(model_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
# Use list slicing to extract every 5th file, starting from the first one
#model_files = model_files[::5]
# Print the resulting list
print(model_files)

########################################################################################################################
#   EXECUTE THE VALIDATION
########################################################################################################################
gan_mae_means = list()
gan_mae_stdevs = list()


time_start = time.time()    # Start the timer

for model_file in model_files:
    generator = os.path.join(path_to_model_to_evaluate, model_file)
    #generator = load_model(model_path)

    # Generate the GAN images
    start_time = time.time()
    gan_images = generate_gan_image(generator, dataset)
    end_time = time.time()

    mae_gan, mae_gan_avg, mae_gan_stddev = compute_errors_gan_only(original_images, gan_images, path_results)

    gan_mae_means.append(mae_gan_avg)
    gan_mae_stdevs.append(mae_gan_stddev)

time_end = time.time()      # End the timer
execution_time = abs(time_start - time_end) # Calculate the run time


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set the font family to "Arial"
rcParams['font.family'] = 'Arial'

# Extract model numbers from model_files for x-axis
generators = [int(file.split('_')[1].split('.')[0].lstrip('0')) for file in model_files]

# Define figure size
plt.figure(figsize=(9, 3))

# Plot the mean MAE
plt.plot(generators, gan_mae_means, marker='o', linestyle='-', color='dimgray', label='Mean MAE')

# Fill the area between (Mean - Std Dev) and (Mean + Std Dev)
plt.fill_between(generators,
                 np.subtract(gan_mae_means, gan_mae_stdevs),
                 np.add(gan_mae_means, gan_mae_stdevs),
                 color='silver', alpha=0.3)

# Set title, labels, and legend
plt.title('SchemaGAN validation over increasing epochs', fontsize=10)
plt.xlabel('Generator epoch', fontsize=10)
plt.ylabel('Mean Absolute error', fontsize=10)
plt.legend(loc='upper right', fontsize=8)

# Rotate x-axis labels for better visibility
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
