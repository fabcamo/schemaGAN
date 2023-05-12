import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import interpolate
from GAN_p2p.functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf

np.random.seed(20232023)

# From DataFusionTools> nearest neighbor interpolation
def nearest_interpolation(training_points, training_data, prediction_points):

    # create KDtree
    tree = cKDTree(training_points)

    # compute closest distance and index of the closest index
    dist, idx = tree.query(prediction_points)
    zn = []
    # create interpolation for every point
    for i in range(len(prediction_points)):
        # interpolate
        zn.append(training_data[idx[i]])

    zn = np.array(zn)

    return zn


########################################################################################################################

# Path for the data
path = 'C:\\inpt\\synthetic_data\\512x32\\validation'

# Define number of rows and columns in 2D grid
no_rows = 32
no_cols = 512
# Create 2D grid with specified number of rows and columns
rows = np.linspace(0, no_rows - 1, no_rows)
cols = np.linspace(0, no_cols - 1, no_cols)
grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)

# Choose missing rate
miss_rate = 0.99
min_distance = 51

########################################################################################################################

# Load the data
all_csv = read_all_csv_files(path)
# Remove data to create fake-CPTs
missing_data, full_data= apply_miss_rate_per_rf(all_csv, miss_rate, min_distance)
no_samples = len(all_csv)
# Reshape the data and store it
missing_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in missing_data])
full_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in full_data])
# Target images> the original synthetic data
tar_images = np.reshape(full_data, (no_samples, no_rows, no_cols, 1))
# Source images> the input "cpt-like" data
src_images = np.reshape(missing_data, (no_samples, no_rows, no_cols, 1))

# Grab the number of validation images
no_validation_images = src_images.shape[0]

# Grab the data from the cpt-like data image (src_image)
coords_all = []     # to store the coordinates
pixel_values_all = []   # to store the pixel values

# Loop over each image in src_images to grab the coordinates with IC values
for i in range(no_validation_images):
    # Get the indices of non-zero values in the i-th image
    # y_indices>[rows] & x_indices>[cols]
    y_indices, x_indices = np.nonzero(src_images[i, :, :, 0])
    # Combine the x and y indices into a 2D array
    # in format> (rows, cols)
    image_coords = np.vstack((y_indices, x_indices)).T
    # Get the pixel values corresponding to the non-zero coordinates
    image_values = src_images[i, y_indices, x_indices, 0]
    # Append the non-zero coordinates to the list
    coords_all.append(image_coords)
    # Append the pixel values to the list
    coord_pix_value = []
    coord_pix_value.extend(image_values.tolist())
    pixel_values_all.append(coord_pix_value)



########################################################################################################################
# Grab the number of validation images
no_validation_images = src_images.shape[0]
nn_results = []
# Loop over each image in src_images to grab the coordinates with IC values
for i in range(no_validation_images):
    # Grab the corresponding coordinates and pixel values for the image at hand
    coords, pixel_values = coords_all[i], pixel_values_all[i]
    # Interpolate onto 2D grid using nearest neighbor interpolation
    #nn_interpolation = []
    nn_interpolation = nearest_interpolation(coords, pixel_values, grid)

    # Reshape the results of a single image to plot
    nn_interpolation = np.reshape(nn_interpolation, (no_rows, no_cols))
    # Append the results to the list
    nn_results.append(nn_interpolation)
########################################################################################################################






########################################################################################################################

    # Plot input data and interpolated output
    fig, axs = plt.subplots(nrows=2, figsize=(10,10))
    # Figure 1> a scatter of the input data (x,y > col,row)
    axs[0].scatter(coords[:,1], coords[:,0], c=pixel_values, marker="v")
    axs[0].set_title('Input data')
    # Figure 2> the interpolated grid
    axs[1].set_title('Interpolated output')
    im = axs[1].imshow(nn_interpolation, extent=[0, no_cols, 0, no_rows], aspect='auto', origin='lower')
    # Set tick labels to be the same for both subplots
    axs[0].set_xticks(np.arange(0, no_cols+1, 50))
    axs[0].set_yticks(np.arange(0, no_rows+1, 5))
    axs[1].set_xticks(np.arange(0, no_cols+1, 50))
    axs[1].set_yticks(np.arange(0, no_rows+1, 5))
    # Invert the axis on the figures
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    # Plot the input pixels on top of the interpolation results as black dots
    axs[1].scatter(coords[:, 1], coords[:, 0], c=pixel_values, edgecolor='k', s=30, marker="v")
    # Show and/or save the plot
    #plt.show()
    #fig.savefig('test_save.png')

########################################################################################################################



    # Calculate the absolute difference
    mae = np.abs(tar_images[i, :, :, 0] - nn_interpolation)
    mae_mean = np.mean(mae)

    # Plot the absolute difference
    plt.imshow(mae, cmap='viridis')
    plt.title(f"Mean absolute error: {round((mae_mean),3)}")
    #plt.show()

print('hola')
