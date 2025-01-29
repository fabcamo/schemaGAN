import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from schemaGAN.functions.utils import IC_normalization
from schemaGAN.functions.summarize import plot_images_error_two_cols, plot_images_error_three_rows


# For local paths
path_full_images = r"D:\schemaGAN\data\BCS\full"  # Path to full images (CSV)
path_missing_images = r"D:\schemaGAN\data\BCS\missing"  # Path to missing images (CSV)
path_bcs_image = r"D:\schemaGAN\data\BCS\bcs_results\bcs_81\bcs_best_estimate.txt"

path_to_model = 'D:\schemaGAN\h5\schemaGAN.h5'
path_results = 'D:/schemaGAN/tests/BCS'

# Generate a random seed using NumPyseed = np.random.randint(20220412, 20230412)
seed = np.random.randint(20220412, 20230412)
# Set the seed for NumPy's random number generator
np.random.seed(seed)

#   CHOOSE THE DIMENSIONS AND MISSING RATE
SIZE_X = 512
SIZE_Y = 32
no_rows = SIZE_Y
no_cols = SIZE_X

def read_bcs_image(file_path):
    # Load the BCS image from the file and reshape it
    data = np.loadtxt(file_path)
    data_reshaped = data[:512, :32].T
    data_reshaped = data_reshaped[::-1, :]  # Flip along x-axis if needed
    return data_reshaped

# Helper function to read CSV files from a folder
def read_csv_files_from_folder(path):
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    images = []
    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        # Swap 'x' and 'z' for correct reshaping:
        # 'x' -> columns (512) and 'z' -> rows (32)
        image_data = df.pivot(index='z', columns='x', values='IC').values.reshape(no_rows, no_cols).astype(np.float32)
        images.append(image_data)
    return np.array(images)

# Function to normalize IC values
def IC_normalization_bcs(images):
    images = np.array(images)
    images = (images - 1) / (4.5 - 1)  # Normalize between 0 and 1
    return images



# Load full data and missing data from CSV files
full_images = read_csv_files_from_folder(path_full_images)
missing_images = read_csv_files_from_folder(path_missing_images)
bcs_image = read_bcs_image(path_bcs_image)

no_samples = len(full_images)

# Reshape the data for neural network compatibility
tar_images = np.reshape(full_images, (no_samples, no_rows, no_cols, 1))
src_images = np.reshape(missing_images, (no_samples, no_rows, no_cols, 1))
bcs_image = np.reshape(bcs_image, (1, no_rows, no_cols, 1))

# Create the array of source and target images
data = [src_images, tar_images]

# Normalize the data from [0 - 4.5] to [-1 to 1]
dataset = IC_normalization(data)
[input_img, orig_img] = dataset

bcs_img = IC_normalization_bcs(bcs_image)  # Ensure it's normalized the same way

# Load the model
model = load_model(path_to_model)

# Create separate lists for gen_image and bcs_image errors
mae_gen_list, mse_gen_list, rmse_gen_list = [], [], []
mae_bcs_list, mse_bcs_list, rmse_bcs_list = [], [], []

for i in range(len(input_img)):
    # Select the cross-section
    ix = np.array([i])
    src_image, tar_image = input_img[ix], orig_img[ix]

    # Generate image using the model
    gen_image = model.predict(src_image)

    # Extract corresponding `bcs_image`
    bcs_image = bcs_img[ix]

    # Compute errors for `gen_image`
    mae_gen = np.mean(np.abs(tar_image - gen_image))
    mse_gen = np.mean(np.square(tar_image - gen_image))
    rmse_gen = np.sqrt(mse_gen)

    # Compute errors for `bcs_image`
    mae_bcs = np.mean(np.abs(tar_image - bcs_image))
    mse_bcs = np.mean(np.square(tar_image - bcs_image))
    rmse_bcs = np.sqrt(mse_bcs)

    # Store errors
    mae_gen_list.append(mae_gen)
    mse_gen_list.append(mse_gen)
    rmse_gen_list.append(rmse_gen)

    mae_bcs_list.append(mae_bcs)
    mse_bcs_list.append(mse_bcs)
    rmse_bcs_list.append(rmse_bcs)

    # Update plotting function to include bcs_image
    plot_images_error_three_rows(src_image, gen_image, bcs_image, tar_image)

    # Save the plot
    plot_acc = os.path.join(path_results, f'plot_acc_{i:06d}.pdf')
    plt.savefig(plot_acc, bbox_inches='tight')
    plt.close()

    print(f">Validation no. {i} completed for model {path_to_model}")