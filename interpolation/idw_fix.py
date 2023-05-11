import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from typing import List, Union
from GAN_p2p.functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf

np.random.seed(20232023)

# From DataFusionTools> nearest neighbor interpolation
def idw_interpolation(training_points, training_data, prediction_points):
    """
    Inverse distance interpolator
    """
    nb_near_points: int = 6
    power: int = 1
    tol: float = 1e-9
    default_cov: float = 10.

    # compute Euclidean distance from grid to training
    tree = cKDTree(training_points)

    # get distances and indexes of the closest nb_points
    dist, idx = tree.query(prediction_points, nb_near_points)
    dist += tol  # to overcome division by zero

    dist = np.array(dist).reshape(nb_near_points)
    idx = np.array(idx).reshape(nb_near_points)

    # for every dataset
    point_aver = []
    point_val = []
    point_var = []
    point_depth = []

    for p in range(nb_near_points):
        # compute the weights
        wei = (1. / dist[p] ** power) / np.sum(1. / dist ** power)
        # if single point
        if point:
            point_aver.append(training_data[idx[p]] * wei)
            point_val.append(training_data[idx[p]])
        # for multiple points
        else:
            point_aver.append(np.log(training_data[idx[p]]) * wei)
            point_val.append(np.log(training_data[idx[p]]))
        point_depth.append(depth_data[idx[p]])

    # compute average
    if point:
        zn = [np.sum(point_aver)]
    else:
        new = []
        for i in range(nb_near_points):
            f = interp1d(point_depth[i], point_aver[i], fill_value=(point_aver[i][-1], point_aver[i][0]),
                         bounds_error=False)
            new.append(f(depth_prediction))
        zn = np.sum(np.array(new), axis=0)

    # compute variance
    if point:
        for p in range(nb_near_points):
            # compute the weighs
            wei = (1. / dist[p] ** power) / np.sum(1. / dist ** power)
            point_var.append((point_val[p] - zn) ** 2 * wei)
    else:

        # compute mean
        new = []
        # 1. for each nearest point p
        for i in range(nb_near_points):
            # 2.  The method first interpolates the training data values onto the prediction depths
            # using linear interpolation.
            f = interp1d(point_depth[i], point_val[i], fill_value=(point_val[i][-1], point_val[i][0]),
                         bounds_error=False)
            new.append(f(depth_prediction))
        # compute variance
        for p in range(nb_near_points):
            # compute the weights
            wei = (1. / dist[p] ** power) / np.sum(1. / dist ** power)
            # compute var
            # 3.  It then computes the squared difference between the interpolated value new[p] and
            # the predicted value zn, and multiplies this squared difference by the weight wei for that
            # training point. This gives the variance contribution for that training point.
            point_var.append((new[p] - zn) ** 2 * wei)
    # 4. The variance of the prediction is the sum of the variance contributions
    # for all the nearest neighbor training points.
    var = np.sum(np.array(point_var), axis=0)

    # add to variables
    if point:
        # update to normal parameters
        zn = zn
        var = var
    else:
        # update to lognormal parameters
        zn = np.exp(zn + var / 2)
        var = np.exp(2 * zn + var) * (np.exp(var) - 1)

    # if only 1 data point is available (var = 0 for all points) -> var is default value
    if nb_near_points == 1:
        var = np.full(len(var), (default_cov * np.array(zn)) ** 2)

    return zn


########################################################################################################################

# Path the the data
path = 'C:\\inpt\\synthetic_data\\test'

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

# Interpolate onto 2D grid using nearest neighbor interpolation
nn_results = idw_interpolation(coords, pixel_values, grid)
nn_results = np.reshape(nn_results,(no_rows, no_cols))


########################################################################################################################

# Plot input data and interpolated output
fig, axs = plt.subplots(nrows=2, figsize=(10,10))
# Figure 1> a scatter of the input data (x,y > col,row)
axs[0].scatter(coords[:,1], coords[:,0], c=pixel_values, marker="v")
axs[0].set_title('Input data')
# Figure 2> the interpolated grid
axs[1].set_title('Interpolated output')
im = axs[1].imshow(nn_results, extent=[0, no_cols, 0, no_rows], aspect='auto', origin='lower')
# Set tick labels to be the same for both subplots
axs[0].set_xticks(np.arange(0, no_cols+1, 50))
axs[0].set_yticks(np.arange(0, no_rows+1, 5))
axs[1].set_xticks(np.arange(0, no_cols+1, 50))
axs[1].set_yticks(np.arange(0, no_rows+1, 5))
# Invert the axis on the figures
axs[0].invert_yaxis()
axs[1].invert_yaxis()
# Plot the input pixels on top of the interpolation results as black dots
axs[1].scatter(coords[:, 1], coords[:, 0], c='black', s=30, marker="v")
# Show and/or save the plot
plt.show()
#fig.savefig('test_save.png')

########################################################################################################################



# Calculate the absolute difference
mae = np.abs(tar_images[0, :, :, 0] - nn_results)
mae_mean = np.mean(mae)

# Plot the absolute difference
plt.imshow(mae, cmap='viridis')
plt.title(f"Mean absolute error: {round((mae_mean),3)}")
plt.show()

