import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from schemaGAN.functions.utils import read_all_csv_files, apply_miss_rate_per_rf, IC_normalization
from schemaGAN.functions.summarize import plot_images_error_two_cols


# For local
path_validation = 'D:/schemaGAN/tests/test4eleni/test_data' # This is CSV data
#path_validation = 'P:/schemagan/synthetic_database/512x32_20k/validation'8 # This is very slow
path_to_model = 'D:/schemaGAN/tests/test4eleni/schemagan_model/model_000036.h5'
path_results = 'D:/schemaGAN/tests/test4eleni/schemagan_inference' # Just were to save the data

# Generate a random seed using NumPyseed = np.random.randint(20220412, 20230412)
seed = np.random.randint(20220412, 20230412)
# Set the seed for NumPy's random number generator
np.random.seed(seed)

#   CHOOSE THE DIMENSIONS AND MISSING RATE
SIZE_X = 512
SIZE_Y = 32
no_rows = SIZE_Y
no_cols = SIZE_X
# Choose missing rate
miss_rate = 0.99
min_distance = 51

# Load the data
all_csv = read_all_csv_files(path_validation)

# Remove data to create fake-CPTs
missing_data, full_data= apply_miss_rate_per_rf(all_csv, miss_rate, min_distance)
no_samples = len(all_csv)

# Reshape the data and store it
missing_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in missing_data])
full_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in full_data])
tar_images = np.reshape(full_data, (no_samples, no_rows, no_cols, 1))
src_images = np.reshape(missing_data, (no_samples, no_rows, no_cols, 1))

# Create the array of source and target images
data = [src_images, tar_images]

# Normalize the data from [0 - 4.5] to [-1 to 1]
dataset = IC_normalization(data)
[input_img, orig_img] = dataset

# Load the model
model = load_model(path_to_model)

# Create the container for the MAE, MSE, RMSE
mae_list = list()
mse_list = list()
rmse_list = list()

for i in range(len(input_img)):
    # Choose a cross-section to run through the generator
    cross_section_number = i
    # Choose a given cross-seciton
    ix = np.array([cross_section_number])
    src_image, tar_image = input_img[ix], orig_img[ix]

    # Generate image from source
    gen_image = model.predict(src_image)

    # Calculate the Mean Absolute Error (MAE) between the target image and the generated one
    mae = np.mean(np.abs(tar_image - gen_image))
    mae_list.append(mae)
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean(np.square(tar_image - gen_image))
    mse_list.append(mse)
    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    rmse_list.append(rmse)

    plot_images_error_two_cols(src_image, gen_image, tar_image)
    # Save the plot as a pdf in the results folder
    plot_acc = os.path.join(path_results, f'plot_acc_{i:06d}.pdf')
    plt.savefig(plot_acc, bbox_inches='tight')

    plt.close()

    print(f">Validation no. {i} completed for model {path_to_model}")