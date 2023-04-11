from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

from GAN_p2p.functions.OLD_p2p2d_model_256 import read_all_csv_files, apply_miss_rate_per_rf, preprocess_data, plot_images
from numpy.random import randn

# Path the the data
path = 'C:\\inpt\\GAN_p2p\\test'

# Images size
SIZE_X = 256
SIZE_Y = 64
no_rows = SIZE_Y
no_cols = SIZE_X

# Choose missing rate
miss_rate = 0.95
min_distance = 10

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

# Plot the original data and the resulting CPT-like data
n_samples = 1
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(src_images[i], cmap='viridis')
# plot target image
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(tar_images[i], cmap='viridis')
plt.show()

# Create the array of source and target images
data = [src_images, tar_images]
# Pre-process the data> normalize
dataset = preprocess_data(data)

# Load the model
model = load_model('C:\\inpt\\GAN_p2p\\results\\r03_256,64_5\\model_050000.h5')

# X1 as input images and X2 as original images
[X1, X2] = dataset

# select random example from all the validation images
ix = np.random.randint(0, len(X1), size=1)
src_image, tar_image = X1[ix], X2[ix]

# generate image from source
gen_image = model.predict(src_image)

# plot all three images> Input, generated and original
plot_images(src_image, gen_image, tar_image)