import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from GAN_p2p.functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf, IC_normalization
from GAN_p2p.interpolation.methods import nearest_interpolation, idw_interpolation, kriging_interpolation
from GAN_p2p.functions.p2p_summary import plot_images_error


########################################################################################################################
#   USER INPUT FOR THE VALIDATION
########################################################################################################################
# Input the name of the generator model to use
name_of_model_to_use = 'model_000200.h5'

# Images size
SIZE_X = 512
SIZE_Y = 32

# Choose missing rate
miss_rate = 0.99
min_distance = 51


########################################################################################################################
#   GENERATE SEED
########################################################################################################################
# Generate a random seed using NumPy
seed = np.random.randint(20220412, 20230412)
# Set the seed for NumPy's random number generator
np.random.seed(seed)


########################################################################################################################
#   PATH FOR THE VALIDATION DATA AND MODEL TO EVALUATE
########################################################################################################################
# For local
path_validation = 'C:\\inpt\\synthetic_data\\512x32\\validation'
path_to_model_to_evaluate = 'C:\\inpt\\GAN_p2p\\results'
path_results = 'C:\\inpt\\GAN_p2p\\results\\test\\validation'


# Iterate over each file in the directory to find the requested model
for filename in os.listdir(path_to_model_to_evaluate):
    # Check if the filename matches the desired name
    if filename == name_of_model_to_use:
        # If we find a matching file, store its full path in the 'generator' variable and exit the loop
        generator = os.path.join(path_to_model_to_evaluate, filename)
        print(f"The '{name_of_model_to_use}' has been selected as the generator")
        break
else:
    # If we don't find a matching file, print a message to the console
    print(f"No file found with name '{name_of_model_to_use}'")


########################################################################################################################
#   CREATE THE GRID
########################################################################################################################
no_rows = SIZE_Y
no_cols = SIZE_X
# Create 2D grid with specified number of rows and columns
rows = np.linspace(0, no_rows - 1, no_rows)
cols = np.linspace(0, no_cols - 1, no_cols)
grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)


########################################################################################################################
#   LOAD THE VALIDATION DATA
########################################################################################################################
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

# Grab the data from the cpt-like data image (src_image)
coord_list = []     # to store the coordinates
pixel_values = []   # to store the pixel values
# Loop over each image in src_images to grab the coordinates with IC values
for i in range(src_images.shape[0]):
    # Get the indices of non-zero values in the i-th image
    # y_indices>[rows] & x_indices>[cols]
    y_indices, x_indices = np.nonzero(src_images[i, :, :, 0])
    # Combine the x and y indices into a 2D array
    # in format> (rows, cols)
    image_coords = np.vstack((y_indices, x_indices)).T
    # Get the pixel values corresponding to the non-zero coordinates
    image_values = src_images[i, y_indices, x_indices, 0]
    # Append the non-zero coordinates to the list
    coord_list.append(image_coords)
    # Append the pixel values to the list
    pixel_values.extend(image_values.tolist())

# Convert the lists to arrays
coords = np.array(coord_list)
coords = coords[0]
pixel_values = np.array(pixel_values)



########################################################################################################################
#   LOAD THE GENERATOR
########################################################################################################################
model = load_model(generator)


########################################################################################################################
#   GENERATE THE GAN INTERPOLATION IMAGES
########################################################################################################################
gan_images = []
for i in range(len(input_img)):
    # Choose a cross-section to run through the generator
    cross_section_number = i
    # Choose a given cross-seciton
    ix = np.array([cross_section_number])
    src_image, tar_image = input_img[ix], orig_img[ix]

    # Generate image from source
    gan_generated_image = model.predict(src_image)
    gan_images.append(gan_generated_image)
    print(f">Generated GAN interpolation no. {i} from model {name_of_model_to_use}")

    # Interpolate onto 2D grid using nearest neighbor interpolation
    nn_results = nearest_interpolation(coords, pixel_values, grid)
    nn_results = np.reshape(nn_results, (no_rows, no_cols))







    plot_images_error(src_image, nn_results, tar_image)
    #validation_dir = os.path.join(path_results, f"validation_{model_file}")
    #if not os.path.exists(validation_dir):
    #    os.mkdir(validation_dir)
    #plot_results_name = os.path.join(validation_dir, f"model_{model_file}_validation_{i}.png")
    plt.show()
    #plt.savefig(plot_results_name)
    plt.close()




















print('Finished')



'''
Load all the original images
Generate the cpt-like images (with a fixed seed)

Load the chosen GENERATOR model
Generate the GAN images from the cpt-like
Store them all in a container> GAN

Call the NN/IDW/Kriging interpolator
From the cpt-like data, generate the images for each one
Store them all in a container> NN, IDW, KRIGING

Create the containers for the error for each method>
MAE_gan, MAE_nn, MAE_idw, MAE_kriging
MSE_gan, MSE_nn, MSE_idw, MSE_kriging
RMSE_gan, RMSE_nn, RMSE_idw, RMSE_kriging

Loop one by one and calculate the error
Store them all in a list for each
Get the average for each
Make some plots comparing them



'''