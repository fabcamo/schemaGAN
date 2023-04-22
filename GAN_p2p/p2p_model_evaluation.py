import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf, IC_normalization
from functions.p2p_summary import plot_images_error

########################################################################################################################
#   PATH FOR THE VALIDATION DATA AND MODEL TO EVALUATE
########################################################################################################################
# For DelftBlue
path_data = r'/scratch/fcamposmontero/databases/512x32/validation'
path_to_model_to_evaluate = r'/scratch/fcamposmontero/results_p2p/512x32_e200_s2000_2/model_000399.h5'
path_results = r'/scratch/fcamposmontero/results_p2p/512x32_e200_s2000_validation'
# For local
#path_validation = 'C:\\inpt\\synthetic_data\\512x32\\validation'
#path_to_model_to_evaluate = 'C:\\inpt\\GAN_p2p\\results\\r07_512x32_5\\model_000200.h5'
#path_results = 'C:\\inpt\\GAN_p2p\\results\\test\\validation'


########################################################################################################################
#   CHOOSE THE DIMENSIONS AND MISSING RATE
########################################################################################################################
# Images size
SIZE_X = 512
SIZE_Y = 32
no_rows = SIZE_Y
no_cols = SIZE_X
# Choose missing rate
miss_rate = 0.90
min_distance = 6

########################################################################################################################

# Load the data
all_csv = read_all_csv_files(path_validation)

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

# Normalize the data from [0 - 4.5] to [-1 to 1]
dataset = IC_normalization(data)
[input_img, orig_img] = dataset

# Load the model
model = load_model(path_to_model_to_evaluate)

# Create the container for the MAE errors
mae_list = list()

for i in range(len(input_img)):
    # Choose a cross-section to run through the generator
    cross_section_number = i
    # Choose a given cross-seciton
    ix = np.array([cross_section_number])
    src_image, tar_image = input_img[ix], orig_img[ix]

    # Generate image from source
    gen_image = model.predict(src_image)

    # Calculate the Mean absolute error between the target image and the generated one
    mae = np.mean(np.absolute(tar_image - gen_image))
    mae_list.append(mae)

    plot_images_error(src_image, gen_image, tar_image)
    plot_results_name = os.path.join(path_results, 'validation_{:06d}.png'.format(i + 1))
    plt.savefig(plot_results_name)
    plt.close()

    print('>Validation no.', i, 'completed')

mae_mean = np.mean(mae_list)

# Save results to dataframe and CSV file
df = pd.DataFrame({'MAE': mae_list})
csv_file = os.path.join(path_results, 'mae_validation.csv')
df.to_csv(csv_file, index=False)
print('>Saved MAE list')


# Generate histogram
hist, bins = np.histogram(mae_list, bins=20)
plt.hist(mae_list, bins=20, color='darkgray')
# Add a vertical line at the mean position
plt.axvline(x=mae_mean, color='dimgray', linestyle='dashed', linewidth=2)
plt.xlabel('Mean absolute error')
plt.ylabel('Frequency')
plt.title(f'Mean absolute error: {mae_mean:.2f}')
# Save the plot
plot_histogram_name = os.path.join(path_results, 'MAE_histogram.png')
plt.savefig(plot_histogram_name)
plt.close()

