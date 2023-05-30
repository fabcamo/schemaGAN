import numpy as np
import matplotlib.pyplot as plt
from GAN_p2p.interpolation.interpolation_utils import generate_idw_images, get_cptlike_data
from GAN_p2p.functions.p2p_process_data import load_remove_reshape_data

np.random.seed(20232023)

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

# load all images, convert to cptlike and reshape them
tar_images, src_images = load_remove_reshape_data(path, miss_rate, min_distance, no_rows, no_cols)

# Get the coordinates and the pixel values for the cotlike data
coords_all, pixel_values_all = get_cptlike_data(src_images)

# Generate Nearest Neighbor images
idw_images = generate_idw_images(no_rows, no_cols, src_images)


########################################################################################################################

coords = coords_all[0]
pixel_values = pixel_values_all[0]
idw_img = idw_images[0]



# Plot input data and interpolated output
fig, axs = plt.subplots(nrows=2, figsize=(10,10))
# Figure 1> a scatter of the input data (x,y > col,row)
axs[0].scatter(coords[:,1], coords[:,0], c=pixel_values, marker="v")
axs[0].set_title('Input data')
# Figure 2> the interpolated grid
axs[1].set_title('Interpolated output')
im = axs[1].imshow(idw_img[0, :, :, 0], extent=[0, no_cols, 0, no_rows], aspect='auto', origin='lower')
# Set tick labels to be the same for both subplots
axs[0].set_xticks(np.arange(0, no_cols+1, 50))
axs[0].set_yticks(np.arange(0, no_rows+1, 5))
axs[1].set_xticks(np.arange(0, no_cols+1, 50))
axs[1].set_yticks(np.arange(0, no_rows+1, 5))
# Invert the axis on the figures
axs[0].invert_yaxis()
axs[1].invert_yaxis()
# Plot the input pixels on top of the interpolation results as black dots
axs[1].scatter(coords[:, 1], coords[:, 0], edgecolor='k', s=30, marker="v")
# Show and/or save the plot
plt.show()
#fig.savefig('test_save.png')



