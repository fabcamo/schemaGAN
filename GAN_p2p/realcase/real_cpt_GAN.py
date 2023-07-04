import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from GAN_p2p.functions.p2p_process_data import IC_normalization, reverse_IC_normalization

from GAN_p2p.interpolation.interpolation_utils import generate_gan_image

# Load the data from the csv files
df1 = pd.read_csv('zeros_A4A18.csv', header=None)
df2 = pd.read_csv('zerosA19A29.csv', header=None)

# Convert the dataframes to numpy arrays
crossec1 = df1.values   # A4 tp A18
crossec2 = df2.values   # A19 to A29

# Reshape them to the format that the other functions know how to handle
crossec1 =crossec1.reshape(1, 32, 512, 1)
crossec2 =crossec2.reshape(1, 32, 512, 1)

# Path to the generator models
path_to_model_to_evaluate = 'C:\\inpt\\GAN_p2p\\results\\test'
# Input the name of the generator model to use
name_of_model_to_use = 'model_000051.h5'


# Generate a random seed using NumPy
seed = np.random.randint(20220412, 20230412)
# Set the seed for NumPy's random number generator
np.random.seed(20232023)


# Pull the Generator model
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

# Dirty way of making the normalization script run
data1 = [crossec1, crossec1]
dataset1 = IC_normalization(data1)
[norm_crossec1, norm_crossec1] = dataset1
data2 = [crossec2, crossec2]
dataset2 = IC_normalization(data2)
[norm_crossec2, norm_crossec2] = dataset2


# Load the generator model from path
model = load_model(generator)

####################################################################################################################
# FOR CS no.1

# Create a subplot for each array
fig, axs = plt.subplots(2, 1, figsize=(10, 5))

# Plot the contents of crossec1 in the first subplot
im1 = axs[0].imshow(crossec1.squeeze(), cmap='viridis')  # using squeeze() to remove singleton dimensions
cbar1 = fig.colorbar(im1, ax=axs[0], orientation='horizontal', fraction=0.08, aspect=40)
cbar1.set_label('Ic values')
axs[0].set_title("CPT input in cross-section A4 - A18")

# Run the GAN
gan_res_crossec1 = model.predict(norm_crossec1)
# Reverse normalization (from [-1,1] to [0,255]) of the generated GAN image
gan_res_crossec1 = reverse_IC_normalization(gan_res_crossec1)
# Remove the singular dimensions
gan_res_crossec1 = np.squeeze(gan_res_crossec1)

# Plot the contents of gan_res_crossec1 in the second subplot
im2 = axs[1].imshow(gan_res_crossec1, cmap='viridis')
cbar2 = fig.colorbar(im2, ax=axs[1], orientation='horizontal', fraction=0.08, aspect=40)
cbar2.set_label('Ic values')
axs[1].set_title("SchemaGAN generated cross-section A4 - A18")

# Automatically adjust subplot parameters to give specified padding
plt.tight_layout()

# Show the plot
plt.show()
plt.clf()

####################################################################################################################
# FOR CS no.2

# Create a subplot for each array
fig, axs = plt.subplots(2, 1, figsize=(10, 5))

# Plot the contents of crossec1 in the first subplot
im1 = axs[0].imshow(crossec2.squeeze(), cmap='viridis')  # using squeeze() to remove singleton dimensions
cbar1 = fig.colorbar(im1, ax=axs[0], orientation='horizontal', fraction=0.08, aspect=40)
cbar1.set_label('Ic values')
axs[0].set_title("CPT input in cross-section A19 - A29")

# Run the GAN
gan_res_crossec2 = model.predict(norm_crossec2)
# Reverse normalization (from [-1,1] to [0,255]) of the generated GAN image
gan_res_crossec2 = reverse_IC_normalization(gan_res_crossec2)
# Remove the singular dimensions
gan_res_crossec2 = np.squeeze(gan_res_crossec2)

# Plot the contents of gan_res_crossec1 in the second subplot
im2 = axs[1].imshow(gan_res_crossec2, cmap='viridis')
cbar2 = fig.colorbar(im2, ax=axs[1], orientation='horizontal', fraction=0.08, aspect=40)
cbar2.set_label('Ic values')
axs[1].set_title("SchemaGAN generated cross-section A19 - A29")

# Automatically adjust subplot parameters to give specified padding
plt.tight_layout()

# Show the plot
plt.show()