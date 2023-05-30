from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

from functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf, IC_normalization
from functions.p2p_summary import plot_images, plot_images_error
from numpy.random import randn

# Path the the data
path = 'C:\\inpt\\synthetic_data\\512x32\\validation'

# Images size
SIZE_X = 512
SIZE_Y = 32
no_rows = SIZE_Y
no_cols = SIZE_X

# Choose missing rate
miss_rate = 0.95
min_distance = 13

# Choose a cross-section to run through the generator
cross_section_number = 2

# Load the data
all_csv = read_all_csv_files(path)
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

# Pre-process the data> normalize
dataset = IC_normalization(data)    # new one

# Load the model
model = load_model('C:\\inpt\\GAN_p2p\\results\\test\\model_000380.h5')

# X1 as input images and X2 as original images
[input_img, orig_img] = dataset

# select random example from all the validation images
#ix = np.random.randint(0, len(X1), size=1)    # Use this one to get a cross-section at random
ix = np.array([cross_section_number])         # Use this one to get a specific cross-section
src_image, tar_image = input_img[ix], orig_img[ix]

# generate image from source
gen_image = model.predict(src_image)

# plot all three images> Input, generated and original
plot_images_error(src_image, gen_image, tar_image)
plt.show()






