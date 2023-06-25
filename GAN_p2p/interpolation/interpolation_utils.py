import os
import csv
import numpy as np
from tensorflow.keras.models import load_model
from GAN_p2p.functions.p2p_process_data import reverse_IC_normalization
from interpolation_methods import nearest_interpolation, idw_interpolation, kriging_interpolation, natural_nei_interpolation, inpt_interpolation



def get_cptlike_data(src_images):
    """
    Extracts the pixel coordinates and corresponding values for non-zero pixels
    in each image from the given list of images.

    Parameters:
    src_images (numpy.ndarray): 4D array of images.

    Returns:
    coords_all (list): List of 2D arrays representing non-zero pixel coordinates in each image.
    pixel_values_all (list): List of pixel values corresponding to the non-zero pixel coordinates in each image.
    """
    coords_all = []         # to store the coordinates
    pixel_values_all = []   # to store the pixel values

    # Loop over each image in src_images to grab the coordinates with IC values
    for i in range(src_images.shape[0]):
        # Get the indices of non-zero values in the i-th image
        y_indices, x_indices = np.nonzero(src_images[i, :, :, 0])
        # Combine the x and y indices into a 2D array (in format: (rows, cols))
        image_coords = np.vstack((y_indices, x_indices)).T
        # Get the pixel values corresponding to the non-zero coordinates
        image_values = src_images[i, y_indices, x_indices, 0]

        # Append the non-zero coordinates to the list
        coords_all.append(image_coords)

        # Append the pixel values to the list
        coord_pix_value = []
        coord_pix_value.extend(image_values.tolist())
        pixel_values_all.append(coord_pix_value)

    return coords_all, pixel_values_all




def format_source_images(dataset):
    """
    Formats the input dataset for plots by reversing the normalization of images.
    The input dataset consists of source and target images.

    Parameters:
    dataset (tuple): A tuple where the first element is an array of source images
                     and the second element is an array of target images.

    Returns:
    original_img (list): A list of target images with reversed normalization.
    cptlike_img (list): A list of source images with reversed normalization.
    """
    # Extract source (input_img) and target (orig_img) images from the dataset
    input_img, orig_img = dataset

    original_img = []
    cptlike_img = []

    # Iterate over each image in the input_img array
    for i in range(input_img.shape[0]):
        # Select an index of a specific cross-section
        cross_section_number = i
        ix = np.array([cross_section_number])

        # Extract the i-th source and target images
        src_image, tar_image = input_img[ix], orig_img[ix]

        # Reverse normalization (from [-1,1] to [0,255]) of the source and target images
        src_image = reverse_IC_normalization(src_image)
        tar_image = reverse_IC_normalization(tar_image)

        # Append the reversed-normalized images to respective lists
        cptlike_img.append(src_image)
        original_img.append(tar_image)

    return original_img, cptlike_img




def generate_gan_image(generator_path, dataset):
    """
    Generates GAN images by passing the source images through the loaded generator model.
    Also, performs reverse normalization on the generated images.

    Parameters:
    generator_path (str): Path to the saved generator model.
    dataset (tuple): A tuple where the first element is an array of source images
                     and the second element is an array of target images.

    Returns:
    gan_images (list): A list of GAN generated images with reversed normalization.
    """
    # Extract source (input_img) and target (orig_img) images from the dataset
    input_img, orig_img = dataset

    # Load the generator model from path
    model = load_model(generator_path)

    gan_images = []

    # Iterate over each image in the input_img array
    for i in range(input_img.shape[0]):
        # Select an index of a specific cross-section
        cross_section_number = i
        ix = np.array([cross_section_number])

        # Extract the i-th source and target images
        src_image, tar_image = input_img[ix], orig_img[ix]

        # Generate the GAN image with the chosen generator
        gan_res = model.predict(src_image)

        # Reverse normalization (from [-1,1] to [0,255]) of the generated GAN image
        gan_res = reverse_IC_normalization(gan_res)

        # Append the reversed-normalized GAN images to the list
        gan_images.append(gan_res)

    return gan_images




def generate_nearnei_images(no_rows, no_cols, src_images):
    """
    Generates images using nearest neighbor interpolation.

    Parameters:
    no_rows (int): Number of rows for the 2D grid for interpolation.
    no_cols (int): Number of columns for the 2D grid for interpolation.
    src_images (numpy.ndarray): 4D array of source images.

    Returns:
    nn_images (list): A list of images generated by nearest neighbor interpolation.
    """

    # Create 2D grid with specified number of rows and columns
    rows = np.linspace(0, no_rows - 1, no_rows)
    cols = np.linspace(0, no_cols - 1, no_cols)
    grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)

    # Get non-zero pixel coordinates and values from the source images
    coords_all, pixel_values_all = get_cptlike_data(src_images)

    nearnei_images = []

    # Iterate over each source image
    for i in range(src_images.shape[0]):
        # Extract the i-th coordinates with pixels and pixel values
        coords, pixel_values = coords_all[i], pixel_values_all[i]

        # Perform nearest neighbor interpolation using the coordinates, pixel values, and grid
        nearnei_res = nearest_interpolation(coords, pixel_values, grid)

        # Reshape the results of a single image to plot
        nearnei_res = np.reshape(nearnei_res, (1, no_rows, no_cols, 1))

        # Append the results to the list
        nearnei_images.append(nearnei_res)

    return nearnei_images




def generate_idw_images(no_rows, no_cols, src_images):
    """
    Generates images using Inverse Distance Weighting (IDW) interpolation.

    Parameters:
    no_rows (int): Number of rows for the 2D grid for interpolation.
    no_cols (int): Number of columns for the 2D grid for interpolation.
    src_images (numpy.ndarray): 4D array of source images.

    Returns:
    idw_images (list): A list of images generated by IDW interpolation.
    """

    # Create 2D grid with specified number of rows and columns
    rows = np.linspace(0, no_rows - 1, no_rows)
    cols = np.linspace(0, no_cols - 1, no_cols)
    grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)

    # Get non-zero pixel coordinates and values from the source images
    coords_all, pixel_values_all = get_cptlike_data(src_images)

    idw_images = []

    # Iterate over each source image
    for i in range(src_images.shape[0]):
        # Extract the i-th coordinates with pixels and pixel values
        coords, pixel_values = coords_all[i], pixel_values_all[i]

        # Perform IDW interpolation using the coordinates, pixel values, and grid
        idw_res = idw_interpolation(coords, pixel_values, grid)
        idw_res = idw_res.T

        # Reshape the results of a single image to plot
        idw_res = np.reshape(idw_res, (1, no_rows, no_cols, 1))

        # Append the results to the list
        idw_images.append(idw_res)

    return idw_images





def generate_krig_images(no_rows, no_cols, src_images):
    """
    Generates images using Kriging interpolation.

    Parameters:
    no_rows (int): Number of rows for the 2D grid for interpolation.
    no_cols (int): Number of columns for the 2D grid for interpolation.
    src_images (numpy.ndarray): 4D array of source images.

    Returns:
    krig_images (list): A list of images generated by Kriging interpolation.
    """

    # Create 2D grid with specified number of rows and columns
    gridx = np.linspace(0, no_cols - 1, no_cols)
    gridy = np.linspace(0, no_rows - 1, no_rows)

    # Get non-zero pixel coordinates and values from the source images
    coords_all, pixel_values_all = get_cptlike_data(src_images)

    krig_images = []

    # Iterate over each source image
    for i in range(src_images.shape[0]):

        # Extract the i-th coordinates with pixels and pixel values
        coords, pixel_values = coords_all[i], pixel_values_all[i]

        # Perform Kriging interpolation using the coordinates, pixel values, and grids
        krig_res = kriging_interpolation(coords, pixel_values, gridx, gridy)

        # Reshape the results of a single image to plot
        krig_res = np.reshape(krig_res, (1, no_rows, no_cols, 1))

        # Append the results to the list
        krig_images.append(krig_res)

    return krig_images




def generate_natnei_images(no_rows, no_cols, src_images):
    """
    Generates images using Natural Neighbor interpolation.

    Parameters:
    no_rows (int): Number of rows for the 2D grid for interpolation.
    no_cols (int): Number of columns for the 2D grid for interpolation.
    src_images (numpy.ndarray): 4D array of source images.

    Returns:
    natnei_images (list): A list of images generated by Natural Neighbor interpolation.
    """

    # Create 2D grid with specified number of rows and columns
    rows = np.linspace(0, no_rows - 1, no_rows)
    cols = np.linspace(0, no_cols - 1, no_cols)
    grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)

    # Get non-zero pixel coordinates and values from the source images
    coords_all, pixel_values_all = get_cptlike_data(src_images)

    natnei_images = []

    # Iterate over each source image
    for i in range(src_images.shape[0]):

        # Extract the i-th coordinates with pixels and pixel values
        coords, pixel_values = coords_all[i], pixel_values_all[i]

        # Perform Natural Neighbor interpolation using the coordinates, pixel values, and grid
        natnei_res = natural_nei_interpolation(coords, pixel_values, grid)

        # Reshape the results of a single image to plot
        natnei_res = np.reshape(natnei_res, (1, no_rows, no_cols, 1))

        # Append the results to the list
        natnei_images.append(natnei_res)

    return natnei_images


def generate_inpainting_images(no_rows, no_cols, src_images):
    """
    Generates images using Inpaiting interpolation.

    Parameters:
    no_rows (int): Number of rows for the 2D grid for interpolation.
    no_cols (int): Number of columns for the 2D grid for interpolation.
    src_images (numpy.ndarray): 4D array of source images.

    Returns:
    idw_images (list): A list of images generated by IDW interpolation.
    """

    # Create 2D grid with specified number of rows and columns
    rows = np.linspace(0, no_rows - 1, no_rows)
    cols = np.linspace(0, no_cols - 1, no_cols)
    grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)

    # Get non-zero pixel coordinates and values from the source images
    coords_all, pixel_values_all = get_cptlike_data(src_images)

    inpt_images = []

    # Iterate over each source image
    for i in range(src_images.shape[0]):
        # Extract the i-th coordinates with pixels and pixel values
        coords, pixel_values = coords_all[i], pixel_values_all[i]

        # Perform IDW interpolation using the coordinates, pixel values, and grid
        inpt_res = inpt_interpolation(coords, pixel_values, grid)

        # Reshape the results of a single image to plot
        inpt_res = np.reshape(inpt_res, (1, no_rows, no_cols, 1))

        # Append the results to the list
        inpt_images.append(inpt_res)

    return inpt_images






def compute_errors(original, gan, nn, idw, krig, natnei, inpt, path):
    """
    This function computes the Mean Absolute Errors (MAE) for different algorithms
    and save the results in a CSV file. It also identifies the indices of minimum
    and maximum values in the GAN MAE list.

    Parameters:
    original (list): Original values.
    gan (list): Generated Adversarial Network's values.
    nn (list): Neural Network's values.
    idw (list): Inverse Distance Weighting's values.
    krig (list): Kriging's values.
    natnei (list): Natural Neighbor's values.
    inpt (list): Inpainting values.

    Returns:
    mae_gan_list (list): MAE list for GAN.
    mae_nn_list (list): MAE list for Neural Network.
    mae_idw_list (list): MAE list for IDW.
    mae_krig_list (list): MAE list for Kriging.
    mae_natnei_list (list): MAE list for Natural Neighbor.
    mae_inpt_list (list): MAE list for Inpainting.
    mae_means (list): Mean of each MAE list.
    """

    # Specify the filename
    filename = 'results_comparison_MAE.csv'
    # Join the path and filename
    full_path = os.path.join(path, filename)

    # Initialize MAE lists for each method
    mae_gan_list = []
    mae_nn_list = []
    mae_idw_list = []
    mae_krig_list = []
    mae_natnei_list = []
    mae_inpt_list = []

    # Calculate MAE for each method and append to the respective list
    for i in range(len(original)):
        mae_gan_list.append(np.mean(np.abs(gan[i] - original[i])))
        mae_nn_list.append(np.mean(np.abs(nn[i] - original[i])))
        mae_idw_list.append(np.mean(np.abs(idw[i] - original[i])))
        mae_krig_list.append(np.mean(np.abs(krig[i] - original[i])))
        mae_natnei_list.append(np.mean(np.abs(natnei[i] - original[i])))
        mae_inpt_list.append(np.mean(np.abs(inpt[i] - original[i])))

    # Calculate the mean of each MAE list and store in mae_means
    mae_means = [np.mean(mae_gan_list), np.mean(mae_nn_list),
                 np.mean(mae_idw_list), np.mean(mae_krig_list),
                 np.mean(mae_natnei_list), np.mean(mae_inpt_list)]

    # Save results to a CSV file at the specified path
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'MAE GAN', 'MAE NearNei', 'MAE IDW', 'MAE Krig', 'MAE NatNei', 'MAE Inpt'])
        for i in range(len(original)):
            writer.writerow([i, mae_gan_list[i], mae_nn_list[i],
                             mae_idw_list[i], mae_krig_list[i], mae_natnei_list[i], mae_inpt_list[i]])

        # Add another row in the CSV with the mean of each MAE list
        writer.writerow(['Mean', mae_means[0], mae_means[1], mae_means[2], mae_means[3], mae_means[4], mae_means[5]])

    # Find the index of the minimum and maximum values in mae_gan_list
    min_index = np.argmin(mae_gan_list)
    max_index = np.argmax(mae_gan_list)

    # Print the index of the minimum and maximum values
    print("Index of minimum value in mae_gan_list:", min_index)
    print("Index of maximum value in mae_gan_list:", max_index)

    return mae_gan_list, mae_nn_list, mae_idw_list, mae_krig_list, mae_natnei_list, mae_inpt_list, mae_means




