import os
import time
import numpy as np
import matplotlib.pyplot as plt


from interpolation_utils import get_cptlike_data, format_source_images, compute_errors
from interpolation_utils import generate_gan_image, generate_nearnei_images, generate_idw_images, generate_krig_images
from interpolation_plots import plot_histograms_row, plot_comparison_of_methods
from GAN_p2p.functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf, IC_normalization


########################################################################################################################
#   USER INPUT FOR THE VALIDATION
########################################################################################################################
# Input the name of the generator model to use
name_of_model_to_use = 'model_000450.h5'

# Images size
SIZE_X = 512
SIZE_Y = 32

# Choose missing rate
miss_rate = 0.99
min_distance = 51

# Choose which validation image to use in the comparison
val_img = 12

########################################################################################################################
#   GENERATE SEED
########################################################################################################################
# Generate a random seed using NumPy
seed = np.random.randint(20220412, 20230412)
# Set the seed for NumPy's random number generator
np.random.seed(20232023)


########################################################################################################################
#   PATH FOR THE VALIDATION DATA AND MODEL TO EVALUATE
########################################################################################################################
# For local validation run
# Path to the validation images
path_validation = 'C:\\inpt\\synthetic_data\\512x32\\validation'
# Path to the generator models
path_to_model_to_evaluate = 'C:\\inpt\\GAN_p2p\\results\\test'
# Path to save the results of the validation
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

# Grab the number of validation images
no_validation_images = src_images.shape[0]

# Create the array of source and target images
data = [src_images, tar_images]

# Normalize the data from [0 - 4.5] to [-1 to 1]
dataset = IC_normalization(data)
[input_img, orig_img] = dataset

# Get the coordinates and the pixel values for the cotlike data
coords_all, pixel_values_all = get_cptlike_data(src_images)

# Format the original image and the cptlike image for ploting
original_images, cptlike_img = format_source_images(dataset)



########################################################################################################################
#   GENERATE THE INTERPOLATION IMAGES
########################################################################################################################
# Generate the GAN images
gan_images = generate_gan_image(generator, dataset)
# Generate Nearest Neighbor images
nearnei_images = generate_nearnei_images(SIZE_Y, SIZE_X, src_images)
# Generate Inverse distance images
idw_images = generate_idw_images(SIZE_Y, SIZE_X, src_images)
# Generate Kriging images
krig_images = generate_krig_images(SIZE_Y, SIZE_X, src_images)
# Generate Natural Neighbor images
natnei_images = generate_krig_images(SIZE_Y, SIZE_X, src_images)


########################################################################################################################
#   CALCULATE THE ERRORS IN THE INTERPOLATION METHODS
########################################################################################################################
# Call for the calculation of the MAE of each interpolation method for each validation image
mae_gan, mae_nn, mae_idw, mae_krig, mae_natnei, mae_means =  compute_errors(
    original_images, gan_images, nearnei_images, idw_images, krig_images, natnei_images, path_results)


########################################################################################################################
#   PLOT THE COMPARISON IMAGES
########################################################################################################################
# Plots three histograms in a row to compare each method with the GAN
plot_histograms_row(mae_gan, mae_nn, mae_idw, mae_natnei)
# Save the plot to the specified path
plt.savefig(os.path.join(path_results, 'histograms_row.png'))
plt.show()


# Plots 10 images for a given validation image, with erros
plot_comparison_of_methods(cptlike_img[val_img], gan_images[val_img],
                           original_images[val_img], nearnei_images[val_img],
                           idw_images[val_img], krig_images[val_img],
                           mae_means)
# Save the plot to the specified path
plt.savefig(os.path.join(path_results, 'comparison_of_methods.png'))
plt.show()


