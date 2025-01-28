import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from schemaGAN.functions.utils import IC_normalization, reverse_IC_normalization
from interpol_compare.functions.utils import generate_nearnei_images, generate_krig_images

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("No GPU available, using CPU.")
else:
    print(f"{len(physical_devices)} GPU(s) available: {physical_devices}")


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Restrict TensorFlow to only use a fraction of the memory
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)




def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: The mean absolute error.
    """
    return abs(y_true - y_pred).mean()



def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: The mean squared error.
    """
    return ((y_true - y_pred) ** 2).mean()



def find_zero_start_row_index(column):
    """
    Find the index of the row where zero starts to appear in a column.

    Parameters:
        column (pandas.Series): The column of data to search.

    Returns:
        int: The index of the row where zero starts to appear.
    """
    zero_start_row = next((i for i, val in enumerate(column) if val == 0), len(column))
    return zero_start_row



def update_dataframe(df, columns_to_check, zero_start_row_indices):
    """
    Update a DataFrame based on the zero start row indices.

    Parameters:
        df (pandas.DataFrame): The DataFrame to update.
        columns_to_check (list): List of column indices to check for zero start.
        zero_start_row_indices (list): List of row indices where zero starts.

    Returns:
        pandas.DataFrame: The updated DataFrame.
    """
    df_copy = df.copy()  # Make a copy of the DataFrame
    for col_idx, start_row_idx in zip(columns_to_check, zero_start_row_indices):
        df_copy.loc[start_row_idx:, col_idx] = 0
    return df_copy



def generate_boxplot(gan, nearnei, krig, method):
    """
    Generate a box plot to compare the Mean Absolute Error (MAE) values
    of different interpol_compare methods.

    Parameters:
        gan (list): List of MAE values for SchemaGAN method.
        nearnei (list): List of MAE values for Nearest Neighbor method.
        krig (list): List of MAE values for Kriging method.
        method (str): The name to use for the Y-axis label.

    Returns:
        None
    """
    # Combine the MAE values into a list of lists
    data = [gan, nearnei, krig]

    # Create a figure and axis object with a specific size
    fig, ax = plt.subplots(figsize=(9, 3))

    # Define the color palette
    color_palette = plt.cm.viridis

    # Create the box plot with filled boxes
    boxplot = ax.boxplot(data, showfliers=True, sym='.', whis=[5, 95], patch_artist=True, showmeans=True, meanline=True)

    # Customize the colors of the boxes
    for i, box in enumerate(boxplot['boxes']):
        # Set the facecolor of the boxes using the Viridis palette
        box.set(facecolor=color_palette(i / len(data)), alpha=0.5)

    # Change the color of the mean line inside the boxes to black
    for median in boxplot['medians']:
        # Set the color of the median line to black
        median.set(color='black', alpha=0)

    # Change the color of the mean line inside the boxes to black
    for mean in boxplot['means']:
        # Set the color of the median line to black
        mean.set(color='black', linestyle='solid')

    # Customize the plot labels
    ax.set_xticklabels(['SchemaGAN', 'Nearest Neighbour', 'Kriging'])

    # Set the y-axis label using the 'method' parameter
    ax.set_ylabel(method)

    # Set the plot title
    ax.set_title('Comparison of Interpolation Methods')



def run_and_get_mean_mae(csv_file):
    """
    Run the interpol_compare methods and calculate the mean MAE for each.

    Parameters:
        csv_file (str): The path to the CSV file containing data.

    Returns:
        tuple: A tuple containing mean MAE for GAN, Near Nei, and Kriging.
    """
    # Load the data from the csv file
    df_all_cpt = pd.read_csv(csv_file, header=None)
    # Print progress message
    #print(f"Data loaded from '{csv_file}'")

    # Remove the first two rows and the first column
    df_all_cpt = df_all_cpt.iloc[2:, 1:].reset_index(drop=True)
    # Update column indexes to start from zero
    df_all_cpt.columns = range(len(df_all_cpt.columns))
    # Convert the entire DataFrame to floats
    df_all_cpt = df_all_cpt.astype(float)
    # Print progress message
    #print("Data cleaned and converted to float")

    # Find the index numbers of columns where the data is different than zero
    cpt_index_locations = df_all_cpt.columns[df_all_cpt.ne(0).any()].tolist()

    # Retry mechanism for selecting 6 CPTs
    max_retries = 25  # Maximum number of retries before failing
    max_retries_per_index = 30  # Maximum retries for finding a valid index
    min_spacing = 50  # Minimum spacing between indices

    for attempt in range(max_retries):
        cpt_index_remaining = []
        retries = 0

        while len(cpt_index_remaining) < 4:
            if retries >= max_retries_per_index:
                print(f"Restarting after {retries} retries (attempt {attempt + 1})")
                break  # Restart the process

            # Randomly pick a candidate index
            selected_index = np.random.choice(cpt_index_locations)

            # Check if the candidate index satisfies the minimum spacing criteria
            if all(abs(selected_index - existing_index) >= min_spacing for existing_index in cpt_index_remaining):
                cpt_index_remaining.append(selected_index)
                retries = 0  # Reset retries after a successful selection
            else:
                retries += 1

        # Print debug info for this attempt
        print(f"Attempt {attempt + 1}: Selected CPTs - {cpt_index_remaining}")

        # Check if we successfully selected 6 CPTs
        if len(cpt_index_remaining) == 5:
            print(f"Successfully selected 5 CPTs on attempt {attempt + 1}")
            break  # Exit retry loop if successful
    else:
        # If retries fail, print a message and raise an exception
        print(f"Failed to select 5 CPTs with spacing >= {min_spacing} after {max_retries} attempts.")
        raise RuntimeError("Unable to select valid CPTs for this run.")

    # Sort the after_removal list in ascending order
    cpt_index_remaining.sort()

    # Create a new list of all the deleted cpt indexes
    # This list of indexes will be used to compare with the original data
    cpt_index_deleted = [cpt for cpt in cpt_index_locations if cpt not in cpt_index_remaining]

    # Create a new DataFrame df_reduced with all zeros BUT the reduced columns remain
    df_reduced = df_all_cpt.copy()
    df_reduced[df_reduced.columns[~df_reduced.columns.isin(cpt_index_remaining)]] = 0.0

    # In order to use the already programmed generator scripts, reshape
    # Convert the dataframes to numpy arrays
    cs_to_evaluate = df_reduced.values.astype(float)  # Convert to float
    # Reshape them to the format that the other functions know how to handle
    cs_to_evaluate = cs_to_evaluate.reshape(1, 32, 512, 1)

    # Dirty way of making the normalization script run
    data_to_norm = [cs_to_evaluate, cs_to_evaluate]
    normalized_data = IC_normalization(data_to_norm)
    [cs_to_evaluate_normalized, cs_to_evaluate_normalized] = normalized_data

    # Generate the images with a random number of CPT and locations given
    # Run the SchemaGAN
    gan_res = model.predict(cs_to_evaluate_normalized)
    #print("SchemaGAN generated image")
    # Reverse normalization (from [-1,1] to [0,255]) of the generated GAN image
    gan_res = reverse_IC_normalization(gan_res)
    # Remove the singular dimensions
    gan_res = np.squeeze(gan_res)
    # Convert to dataframe and save it
    df_gan = pd.DataFrame(gan_res)
    # Run the Nearest Neighbour interpol_compare
    nearnei_res = generate_nearnei_images(SIZE_Y, SIZE_X, cs_to_evaluate)
    nearnei_res = np.squeeze(nearnei_res)
    # Convert to dataframe and save it
    df_nn = pd.DataFrame(nearnei_res)
    #print("Nearest Neighbour generated image")
    # Run the Kriging interpol_compare
    krig_res = generate_krig_images(SIZE_Y, SIZE_X, cs_to_evaluate)
    krig_res = np.squeeze(krig_res)
    # Convert to dataframe and save it
    df_krig = pd.DataFrame(krig_res)
    #print("Kriging generated image")

    # Find the index of the row where zero starts to appear in the specified columns and save them in a list
    zero_start_rows_per_column = df_all_cpt[df_all_cpt.columns[cpt_index_deleted]].apply(find_zero_start_row_index).tolist()

    # Update the second DataFrame based on the zero_start_rows_per_column list
    df_gan_to_compare = update_dataframe(df_gan, cpt_index_deleted, zero_start_rows_per_column)
    df_near_to_compare = update_dataframe(df_nn, cpt_index_deleted, zero_start_rows_per_column)
    df_krig_to_compare = update_dataframe(df_krig, cpt_index_deleted, zero_start_rows_per_column)

    # Create empty dictionaries to hold the MAE
    mae_gan_results = {}
    mae_near_results = {}
    mae_krig_results = {}
    # Loop through the columns deleted from the original to compare
    for index in cpt_index_deleted:
        mae_gan_results[index] = mean_absolute_error(df_all_cpt[index], df_gan_to_compare[index])
        mae_near_results[index] = mean_absolute_error(df_all_cpt[index], df_near_to_compare[index])
        mae_krig_results[index] = mean_absolute_error(df_all_cpt[index], df_krig_to_compare[index])

    # Compute the mean of all the MAEs
    mean_mae_gan = sum(mae_gan_results.values()) / len(mae_gan_results)
    mean_mae_nearnei = sum(mae_near_results.values()) / len(mae_near_results)
    mean_mae_kriging = sum(mae_krig_results.values()) / len(mae_krig_results)
    #print(f"Mean MAE for SchemaGAN: {mean_mae_gan}")
    #print(f"Mean MAE for Nearest Neighbour: {mean_mae_nearnei}")
    #print(f"Mean MAE for Kriging: {mean_mae_kriging}")

    # Create empty dictionaries to hold the MSE
    mse_gan_results = {}
    mse_near_results = {}
    mse_krig_results = {}

    # Loop through the columns deleted from the original to compare
    for index in cpt_index_deleted:
        mse_gan_results[index] = mean_squared_error(df_all_cpt[index], df_gan_to_compare[index])
        mse_near_results[index] = mean_squared_error(df_all_cpt[index], df_near_to_compare[index])
        mse_krig_results[index] = mean_squared_error(df_all_cpt[index], df_krig_to_compare[index])

    # Compute the mean of all the MSEs
    mean_mse_gan = sum(mse_gan_results.values()) / len(mse_gan_results)
    mean_mse_nearnei = sum(mse_near_results.values()) / len(mse_near_results)
    mean_mse_kriging = sum(mse_krig_results.values()) / len(mse_krig_results)
    #print(f"Mean MSE for SchemaGAN: {mean_mse_gan}")
    #print(f"Mean MSE for Nearest Neighbour: {mean_mse_nearnei}")
    #print(f"Mean MSE for Kriging: {mean_mse_kriging}")

    return mean_mae_gan, mean_mae_nearnei, mean_mae_kriging, mean_mse_gan, mean_mse_nearnei, mean_mse_kriging


#############################################################################################################
if __name__ == "__main__":

    # Images size
    SIZE_X = 512
    SIZE_Y = 32

    # Number of runs
    num_runs = 1000

    # FOR THE SCHEMAGAN MODEL
    # Path to the generator models
    path_to_model_to_evaluate = 'D:\schemaGAN\h5'
    # Input the name of the generator model to use
    name_of_model_to_use = 'schemaGAN.h5'

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

    # Load the generator model from path
    model = load_model(generator)

    # Lists to store the average MAE results from each run
    avg_mae_gan = []
    avg_mae_near = []
    avg_mae_krig = []
    # Lists to store the average MSE results from each run
    avg_mse_gan = []
    avg_mse_near = []
    avg_mse_krig = []

    # Perform num_runs runs for EINDHOVEN DATA #######################################################################
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        mean_mae_gan, mean_mae_near, mean_mae_krig, mean_mse_gan, mean_mse_near, mean_mse_krig = run_and_get_mean_mae("D:/schemaGAN/data/eindhoven/eind01_512x32.csv")

        avg_mae_gan.append(mean_mae_gan)
        avg_mae_near.append(mean_mae_near)
        avg_mae_krig.append(mean_mae_krig)
        avg_mse_gan.append(mean_mse_gan)
        avg_mse_near.append(mean_mse_near)
        avg_mse_krig.append(mean_mse_krig)

        # print progress message
        #print(f"Run {run + 1}/{num_runs} completed for first file")

        mean_mae_gan, mean_mae_near, mean_mae_krig, mean_mse_gan, mean_mse_near, mean_mse_krig = run_and_get_mean_mae("D:/schemaGAN/data/eindhoven/eind02_512x32.csv")
        avg_mae_gan.append(mean_mae_gan)
        avg_mae_near.append(mean_mae_near)
        avg_mae_krig.append(mean_mae_krig)
        avg_mse_gan.append(mean_mse_gan)
        avg_mse_near.append(mean_mse_near)
        avg_mse_krig.append(mean_mse_krig)

        # print progress message
        #print(f"Run {run + 1}/{num_runs} completed for second file")


    # Save the average MAE results to a CSV file
    df_average_mae = pd.DataFrame({
        "Average_MAE_GAN": avg_mae_gan,
        "Average_MAE_NEAR": avg_mae_near,
        "Average_MAE_KRIG": avg_mae_krig,
        "Average_MSE_GAN": avg_mse_gan,
        "Average_MSE_NEAR": avg_mse_near,
        "Average_MSE_KRIG": avg_mse_krig
    })
    df_average_mae.to_csv("D:/schemaGAN/real_case/eindhoven/average_mae_results.csv", index=False)

    # Calculate the total average MAE for each method
    total_average_mae_gan = np.mean(avg_mae_gan)
    total_average_mae_near = np.mean(avg_mae_near)
    total_average_mae_krig = np.mean(avg_mae_krig)
    # Calculate the total average MSE for each method
    total_average_mse_gan = np.mean(avg_mae_gan)
    total_average_mse_near = np.mean(avg_mae_near)
    total_average_mse_krig = np.mean(avg_mae_krig)

    # Plot the histogram of MAE errors
    generate_boxplot(avg_mae_gan, avg_mae_near, avg_mae_krig, method='Mean absolute error')
    # Save the plot to the specified path
    plt.savefig(os.path.join('D:/schemaGAN/real_case/eindhoven/boxplot_mae.pdf'), format='pdf')
    plt.close()

    # Plot the histogram of MSE errors
    generate_boxplot(avg_mse_gan, avg_mse_near, avg_mse_krig, method='Mean squared error')
    # Save the plot to the specified path
    plt.savefig(os.path.join('D:/schemaGAN/real_case/eindhoven/boxplot_mse.pdf'), format='pdf')
    plt.close()

    ######################## GRONINGEN ############################################################################################################
    # Number of runs
    num_runs = 1000
    # Lists to store the average MAE results from each run
    avg_mae_gan = []
    avg_mae_near = []
    avg_mae_krig = []
    # Lists to store the average MSE results from each run
    avg_mse_gan = []
    avg_mse_near = []
    avg_mse_krig = []

    # Perform num_runs runs for EINDHOVEN DATA #######################################################################
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        mean_mae_gan, mean_mae_near, mean_mae_krig, mean_mse_gan, mean_mse_near, mean_mse_krig = run_and_get_mean_mae("D:/schemaGAN/data/groningen/gro01_512x32.csv")

        avg_mae_gan.append(mean_mae_gan)
        avg_mae_near.append(mean_mae_near)
        avg_mae_krig.append(mean_mae_krig)
        avg_mse_gan.append(mean_mse_gan)
        avg_mse_near.append(mean_mse_near)
        avg_mse_krig.append(mean_mse_krig)

        # print progress message
        #print(f"Run {run + 1}/{num_runs} completed for first file")

        mean_mae_gan, mean_mae_near, mean_mae_krig, mean_mse_gan, mean_mse_near, mean_mse_krig = run_and_get_mean_mae("D:/schemaGAN/data/groningen/gro02_512x32.csv")
        avg_mae_gan.append(mean_mae_gan)
        avg_mae_near.append(mean_mae_near)
        avg_mae_krig.append(mean_mae_krig)
        avg_mse_gan.append(mean_mse_gan)
        avg_mse_near.append(mean_mse_near)
        avg_mse_krig.append(mean_mse_krig)

        # print progress message
        #print(f"Run {run + 1}/{num_runs} completed for second file")

        mean_mae_gan, mean_mae_near, mean_mae_krig, mean_mse_gan, mean_mse_near, mean_mse_krig = run_and_get_mean_mae("D:/schemaGAN/data/groningen/gro03_512x32.csv")
        avg_mae_gan.append(mean_mae_gan)
        avg_mae_near.append(mean_mae_near)
        avg_mae_krig.append(mean_mae_krig)
        avg_mse_gan.append(mean_mse_gan)
        avg_mse_near.append(mean_mse_near)
        avg_mse_krig.append(mean_mse_krig)

        # print progress message
        #print(f"Run {run + 1}/{num_runs} completed for second file")


    # Save the average MAE results to a CSV file
    df_average_mae = pd.DataFrame({
        "Average_MAE_GAN": avg_mae_gan,
        "Average_MAE_NEAR": avg_mae_near,
        "Average_MAE_KRIG": avg_mae_krig,
        "Average_MSE_GAN": avg_mse_gan,
        "Average_MSE_NEAR": avg_mse_near,
        "Average_MSE_KRIG": avg_mse_krig
    })
    df_average_mae.to_csv("D:/schemaGAN/real_case/groningen/average_mae_results.csv", index=False)

    # Calculate the total average MAE for each method
    total_average_mae_gan = np.mean(avg_mae_gan)
    total_average_mae_near = np.mean(avg_mae_near)
    total_average_mae_krig = np.mean(avg_mae_krig)
    # Calculate the total average MSE for each method
    total_average_mse_gan = np.mean(avg_mae_gan)
    total_average_mse_near = np.mean(avg_mae_near)
    total_average_mse_krig = np.mean(avg_mae_krig)

    # Plot the histogram of MAE errors
    generate_boxplot(avg_mae_gan, avg_mae_near, avg_mae_krig, method='Mean absolute error')
    # Save the plot to the specified path
    plt.savefig(os.path.join('D:/schemaGAN/real_case/groningen/boxplot_mae.pdf'), format='pdf')
    plt.close()

    # Plot the histogram of MSE errors
    generate_boxplot(avg_mse_gan, avg_mse_near, avg_mse_krig, method='Mean squared error')
    # Save the plot to the specified path
    plt.savefig(os.path.join('D:/schemaGAN/real_case/groningen/boxplot_mse.pdf'), format='pdf')
    plt.close()

    # ####################################################################################################################
    # # FOR CS no.1
    #
    # # Create a subplot for each array
    # fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    #
    # # Plot the contents of crossec1 in the first subplot
    # im1 = axs[0].imshow(cs_to_evaluate.squeeze(), cmap='viridis')  # using squeeze() to remove singleton dimensions
    # cbar1 = fig.colorbar(im1, ax=axs[0], orientation='horizontal', fraction=0.08, aspect=40)
    # cbar1.set_label('Ic values')
    # axs[0].set_title("CPT input in cross-section A4 - A18")
    #
    #
    # # Plot the contents of gan_res_crossec1 in the second subplot
    # im2 = axs[1].imshow(gan_res, cmap='viridis')
    # cbar2 = fig.colorbar(im2, ax=axs[1], orientation='horizontal', fraction=0.08, aspect=40)
    # cbar2.set_label('Ic values')
    # axs[1].set_title("SchemaGAN generated cross-section A4 - A18")
    #
    # # Plot black lines at the CPT indexes in cpt_index_deleted
    # for cpt_idx in cpt_index_deleted:
    #     axs[1].axvline(cpt_idx, color='black', linestyle='dotted', linewidth=2)
    #
    # # Automatically adjust subplot parameters to give specified padding
    # plt.tight_layout()
    #
    # # Automatically adjust subplot parameters to give specified padding
    # plt.tight_layout()
    #
    # # Show the plot
    # plt.show()
    # plt.clf()

