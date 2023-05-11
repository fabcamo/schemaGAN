import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from GAN_p2p.functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf

# Path the the data
path = 'C:\\inpt\\synthetic_data\\test'

# Define number of rows and columns in 2D grid
no_rows = 32
no_cols = 512

# Choose missing rate
miss_rate = 0.99
min_distance = 51

# Load the data
all_csv = read_all_csv_files(path)

# Remove data to create fake-CPTs
missing_data, full_data= apply_miss_rate_per_rf(all_csv, miss_rate, min_distance)

cptlike = missing_data[0]

# This does not work, allthough non-nan are still inside of cptlike, there are not recognized.
#cptlike = np.where(cptlike == 0.00000, np.nan, cptlike)

# This does not work either,
cptlike[cptlike == 0] = np.nan

#new = np.ma.masked_equal(cptlike, 0.00000)



# Plot the masked array with imshow
plt.imshow(cptlike, s=20, cmap='viridis')



# Add a colorbar to the plot
plt.colorbar()

plt.show()