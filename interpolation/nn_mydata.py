import matplotlib.pyplot as plt
import numpy as np

from GAN_p2p.functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf, preprocess_data, IC_normalization


# Path the the data
path = 'C:\\inpt\\synthetic_data\\test'

# Images size
SIZE_X = 512
SIZE_Y = 32
no_rows = SIZE_Y
no_cols = SIZE_X

# Generating 1D arrays of X and Y values
X = np.linspace(0, SIZE_X-1, SIZE_X)
Y = np.linspace(0, SIZE_Y-1, SIZE_Y)
# Creating a 2D grid using X and Y values
X, Y = np.meshgrid(X, Y)

# Choose missing rate
miss_rate = 0.99
min_distance = 51

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











# Create a figure with a size of 10 inches by 4 inches
plt.imshow(src_images[0, :, :, 0], cmap='viridis')
plt.show()