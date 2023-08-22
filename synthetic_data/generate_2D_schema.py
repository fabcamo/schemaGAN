import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions.layering import generate_synthetic
from functions.utils import split_data, save_summary


# Check the time and start the timers
time_current = time.strftime("%d/%m/%Y %H:%M:%S")

########################################################################################################################
#   GENERATE SEED
########################################################################################################################
# Generate a random seed using NumPy
seed = np.random.randint(20220412, 20230412)
# Set the seed for NumPys random number generator
np.random.seed(seed)


########################################################################################################################
#   USER INPUT FOR TRAINING
########################################################################################################################
output_folder = 'C:\\inpt\\synthetic_data\\512x32'  # Choose the output folder
no_realizations = 20     # Number of realizations to generate
vali_ratio = 0.1        # Percentage of total data for validation
test_ratio = 0.1        # Percentage of total data for testing
x_max = 512     # Length (x) of the model
z_max = 32      # Depth (z) of the model


########################################################################################################################
#   GEOMETRY PRE-PROCESS
########################################################################################################################
x_coord = np.arange(0, x_max, 1)       # array of x coordinates
z_coord = np.arange(0, z_max, 1)       # array of z coordinates
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")   # 2D mesh of coordinates x,z


########################################################################################################################
#   GENERATE THE SYNTHETIC DATA
########################################################################################################################
# Start the timer
time_start = time.time()
counter = 0
while counter < no_realizations:
    try:
        print('Generating model no.:', counter+1)
        generate_synthetic(output_folder, counter, z_max, x_max, seed)
        counter += 1

    except Exception as e:
        print(f"Error in generating model no. {counter + 1}: {e}")
        continue

##### SPLIT THE DATA INTO TRAINING AND VALIDATION AT RANDOM ############################################################
validation_folder = os.path.join(output_folder, "validation")
test_folder = os.path.join(output_folder, "test")

# Split the data generated into training, validation and testing
split_data(output_folder, os.path.join(output_folder, "train"), validation_folder, test_folder, vali_ratio, test_ratio)


# End the timer
time_end = time.time()

# Save a summary of the run times and seed
save_summary(output_folder, time_start, time_end, seed, no_realizations)
