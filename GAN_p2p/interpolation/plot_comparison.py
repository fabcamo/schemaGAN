import numpy as np
import matplotlib.pyplot as plt
from GAN_p2p.functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf


########################################################################################################################
#   USER INPUT FOR THE VALIDATION
########################################################################################################################
# Fix the seed for the validation
seed = np.random.randint(20220412, 20230412)
# Set the seed for NumPy's random number generator
np.random.seed(seed)

# Path to the validation data
path = '/synthetic_data/test'

# Define number of rows and columns in 2D grid
no_rows = 32
no_cols = 512

# Choose missing rate
miss_rate = 0.99
min_distance = 51


########################################################################################################################
#   LOAD THE VALIDATION DATA
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


########################################################################################################################
#   USER INPUT FOR THE VALIDATION
########################################################################################################################
def plot_comparison_of_methods(src_img, gen_img, tar_img, nn, idw, kriging):

    # Stack all the images
    images = np.vstack((tar_img, src_img,
                        tar_img, np.abs(src_img-tar_img),
                        tar_img, np.abs(src_img-tar_img),
                        tar_img, np.abs(src_img-tar_img),
                        tar_img, np.abs(src_img-tar_img)))

    # Set plot titles for each subplot
    titles = ['Original cross-section', 'Input CPT data',
              'GAN prediction', 'GAN MAE:',
              'Nearest neighbor interpolation', 'Nearest neighbor MAE:',
              'Inverse distance interpolation', 'Inverse distance MAE:',
              'Ordinary kriging interpolation', 'Ordinary kriging MAE:']

    # Set the axis labels for each subplot
    xlabels = ['', '', '', '', '', '', '', '', 'Distance', 'Distance']
    ylabels = ['Depth', '', 'Depth', '', 'Depth', '', 'Depth', '', 'Depth', '']

    # Set the cbar range for each subplot
    ranges_vmin_vmax = [[1, 4.5], [0, 4.5],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1]]

    # Set the cbar titles for each subplot
    cbar_titles = ['', '', '', '', '', '', '', '', 'IC values', 'IC error values']

    # Defining the number of subplots to use in the overall figure
    num_images = len(images)
    num_rows = 5
    num_cols = int(np.ceil(num_images / num_rows))

    # Create a figure with a size of 8 inches by 5 inches
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 5))

    # Plot images row by row
    for i in range(num_images):
        # Define subplot
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        im = ax.imshow(images[i, :, :, 0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])

        # Set title with fontsize
        ax.set_title(titles[i], fontsize=8)

        # Set tick_params with fontsize
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=7)

        # Set x and y labels
        ax.set_xlabel(xlabels[i], fontsize=7)
        ax.set_ylabel(ylabels[i], fontsize=7)

        # Manually set tick mark spacing
        ax.set_xticks(np.arange(0, images.shape[2], 50))
        ax.set_yticks(np.arange(0, images.shape[1], 15))

        # Set the size of the subplot
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([0, 512])
        ax.set_ylim([32, 0])

        # Add colorbar only to last row
        if row == num_rows - 1:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.08, pad=0.5, aspect=40)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(cbar_titles[i], fontsize=7)
            ax.set_xlim([0, 512])
            ax.set_ylim([32, 0])

    plt.tight_layout()


