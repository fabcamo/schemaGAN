import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time



# Set the font family to "Arial"
rcParams['font.family'] = 'Arial'


from interpolation_utils import get_cptlike_data, format_source_images, compute_errors
from interpolation_utils import generate_gan_image, generate_nearnei_images, generate_idw_images, generate_krig_images, generate_inpainting_images, generate_natnei_images
from interpolation_plots import plot_histograms_row, plot_comparison_of_methods
from GAN_p2p.functions.p2p_process_data import load_remove_reshape_data, IC_normalization


########################################################################################################################
#   USER INPUT FOR THE VALIDATION
########################################################################################################################
# Input the name of the generator model to use
name_of_model_to_use = 'model_000051.h5'

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
path_results = 'C:\\inpt\\GAN_p2p\\results\\test\\validation_new'
# Specify the output file
output_file = 'C:\\inpt\\GAN_p2p\\results\\test\\validation_new\\times.txt'


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

# Save seed to the text file
with open(output_file, 'a') as f:
    f.write(f"Seed number used: {seed}\n")


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
#   GENERATE THE INTERPOLATION IMAGES
########################################################################################################################


# Generate the GAN images
start_time = time.time()
gan_images = generate_gan_image(generator, dataset)
end_time = time.time()
with open(output_file, 'a') as f:
    f.write("GAN images generation took {:.2f} seconds.\n".format(end_time - start_time))

# Generate Nearest Neighbor images
start_time = time.time()
nearnei_images = generate_nearnei_images(SIZE_Y, SIZE_X, src_images)
end_time = time.time()
with open(output_file, 'a') as f:
    f.write("Nearest Neighbor images generation took {:.2f} seconds.\n".format(end_time - start_time))

# Generate Inverse distance images
start_time = time.time()
idw_images = generate_idw_images(SIZE_Y, SIZE_X, src_images)
end_time = time.time()
with open(output_file, 'a') as f:
    f.write("Inverse distance images generation took {:.2f} seconds.\n".format(end_time - start_time))

# Generate Kriging images
start_time = time.time()
krig_images = generate_krig_images(SIZE_Y, SIZE_X, src_images)
end_time = time.time()
with open(output_file, 'a') as f:
    f.write("Kriging images generation took {:.2f} seconds.\n".format(end_time - start_time))

# Generate Natural Neighbor images
start_time = time.time()
natnei_images = generate_natnei_images(SIZE_Y, SIZE_X, src_images)
end_time = time.time()
with open(output_file, 'a') as f:
    f.write("Natural Neighbor images generation took {:.2f} seconds.\n".format(end_time - start_time))

# Generate Inpainting images
start_time = time.time()
inpt_images = generate_inpainting_images(SIZE_Y, SIZE_X, src_images)
end_time = time.time()
with open(output_file, 'a') as f:
    f.write("Inpainting images generation took {:.2f} seconds.\n".format(end_time - start_time))



########################################################################################################################
#   CALCULATE THE ERRORS IN THE INTERPOLATION METHODS
########################################################################################################################
# Call for the calculation of the MAE of each interpolation method for each validation image
mae_gan, mae_nn, mae_idw, mae_krig, mae_natnei, mae_inpt, mae_means =  compute_errors(
    original_images, gan_images, nearnei_images, idw_images, krig_images, natnei_images, inpt_images, path_results)


########################################################################################################################
#   PLOT THE COMPARISON IMAGES
########################################################################################################################
# Plots three histograms in a row to compare each method with the GAN
plot_histograms_row(mae_gan, mae_nn, mae_idw, mae_krig, mae_natnei, mae_inpt)
# Save the plot to the specified path
plt.savefig(os.path.join(path_results, 'histograms_row.pdf'), format='pdf')
plt.show()



for i in range(no_validation_images):
    # Get the validation image MAE for each method
    val_img = [mae_gan[i], mae_nn[i], mae_idw[i], mae_krig[i], mae_natnei[i], mae_inpt[i]]

    # Plots 10 images for a given validation image, with erros
    plot_comparison_of_methods(cptlike_img[i], gan_images[i],
                               original_images[i], nearnei_images[i],
                               idw_images[i], krig_images[i],
                               natnei_images[i], inpt_images[i], val_img)
    # Save the plot to the specified path with a dynamic figure name
    plt.savefig(os.path.join(path_results, f'compar_{i}_of_4000.pdf'), format='pdf')

