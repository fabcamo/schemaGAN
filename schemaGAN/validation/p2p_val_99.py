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
path_validation = r'/scratch/fcamposmontero/databases/512x32_20k/validation'
path_to_model_to_evaluate = r'/scratch/fcamposmontero/results_p2p/512x32_e200_s16k'
path_results = r'/scratch/fcamposmontero/results_p2p/512x32_e200_s16k/validation'
# For local
#path_validation = 'C:\\inpt\\synthetic_data\\512x32\\validation'
#path_to_model_to_evaluate = 'C:\\inpt\\schemaGAN\\results\\test'
#path_results = 'C:\\inpt\\schemaGAN\\results\\test\\validation'


########################################################################################################################
#   CHOOSE THE DIMENSIONS AND MISSING RATE
########################################################################################################################
# Images size
SIZE_X = 512
SIZE_Y = 32
no_rows = SIZE_Y
no_cols = SIZE_X
# Choose missing rate
miss_rate = 0.99
min_distance = 51

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

########################################################################################################################
# Create an empty dataframe to store the mean MAE for each model
mae_mean_list = list()
mse_mean_list = list()
rmse_mean_list = list()

# Use os.listdir to get a list of all the files in the directory
all_files = os.listdir(path_to_model_to_evaluate)
# Use list comprehension to filter out only the files that end with '.h5'
model_files = [file for file in all_files if file.endswith('.h5')]
# Use a lambda function to extract the XXXXXX part of the filename and sort the files by it
model_files = sorted(model_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
# Use list slicing to extract every 5th file, starting from the first one
model_files = model_files[::5]
# Print the resulting list
print(model_files)


for model_file in model_files:
    model_path = os.path.join(path_to_model_to_evaluate, model_file)
    model = load_model(model_path)

    # Create the container for the MAE, MSE, RMSE
    mae_list = list()
    mse_list = list()
    rmse_list = list()

    for i in range(len(input_img)):
        # Choose a cross-section to run through the generator
        cross_section_number = i
        # Choose a given cross-seciton
        ix = np.array([cross_section_number])
        src_image, tar_image = input_img[ix], orig_img[ix]

        # Generate image from source
        gen_image = model.predict(src_image)

        # Calculate the Mean Absolute Error (MAE) between the target image and the generated one
        mae = np.mean(np.abs(tar_image - gen_image))
        mae_list.append(mae)
        # Calculate the Mean Squared Error (MSE)
        mse = np.mean(np.square(tar_image - gen_image))
        mse_list.append(mse)
        # Calculate the Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)

        plot_images_error(src_image, gen_image, tar_image)
        validation_dir = os.path.join(path_results, f"validation_{model_file}")
        if not os.path.exists(validation_dir):
            os.mkdir(validation_dir)
        plot_results_name = os.path.join(validation_dir, f"model_{model_file}_validation_{i}.png")
        plt.savefig(plot_results_name)
        plt.close()

        print(f">Validation no. {i} completed for model {model_path}")

    # Save results to dataframe and CSV file
    df = pd.DataFrame({'MAE': mae_list, 'MSE': mse_list, 'RMSE': rmse_list})
    validation_csv = os.path.join(path_results, f"validation_{model_file}.csv")
    df.to_csv(validation_csv, index=False)
    print('>Saved validation list')

    # Calculate the mean MAE for the model
    mae_mean = np.mean(mae_list)
    mse_mean = np.mean(mse_list)
    rmse_mean = np.mean(rmse_list)
    # Add the mean MAE to the dataframe
    mae_mean_list.append(mae_mean)
    mse_mean_list.append(mse_mean)
    rmse_mean_list.append(rmse_mean)

    # Generate histogram
    hist, bins = np.histogram(mae_list, bins=20)
    plt.hist(mae_list, bins=20, color='darkgray')
    # Add a vertical line at the mean position
    plt.axvline(x=mae_mean, color='dimgray', linestyle='dashed', linewidth=2)
    plt.xlabel('Mean absolute error')
    plt.ylabel('Frequency')
    plt.title(f'Mean absolute error: {mae_mean:.2f}')
    # Save the plot
    plot_histogram_name = os.path.join(path_results, f"MAE_histogram_{model_file}.png")
    plt.savefig(plot_histogram_name)
    plt.close()




# Save the results to a CSV file with the name "validation_means_{i}.csv"
validation_csv = os.path.join(path_results, f"means_{model_file}.csv")
df = pd.DataFrame({'MAE_mean': mae_mean_list, 'MSE_mean': mse_mean_list, 'RMSE_mean': rmse_mean_list})
df.to_csv(validation_csv, index=False)
print('>Saved Means list')

# Plot the validation error for all models/epochs
plt.figure(figsize=(8, 3))  # set figure size as 8x3 inches
plt.plot(df.index, df['MAE_mean'], color='black', label='MAE Mean')
plt.plot(df.index, df['MSE_mean'], color='gray', label='MSE Mean')
plt.plot(df.index, df['RMSE_mean'], color='lightgray', label='RMSE Mean')
plt.xlabel('Epochs')
plt.ylabel('Error statistics')
plt.legend(loc='upper right')
plt.savefig(os.path.join(path_results, 'validation_error.png'), dpi=300, bbox_inches='tight')
#plt.show()














