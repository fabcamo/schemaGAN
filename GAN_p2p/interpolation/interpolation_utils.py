import os
import csv
import numpy as np
from skimage import metrics
from matplotlib import pyplot as plt

from tensorflow.keras.models import load_model
from GAN_p2p.functions.p2p_process_data import reverse_IC_normalization
from GAN_p2p.interpolation.interpolation_methods import nearest_interpolation, idw_interpolation, kriging_interpolation, natural_nei_interpolation, inpt_interpolation


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





def compute_mae(original, gan, nn, idw, krig, natnei, inpt, path):
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




def compute_mse(original, gan, nn, idw, krig, natnei, inpt, path):
    """
    This function computes the Mean Squared Errors (MSE) for different algorithms
    and save the results in a CSV file. It also identifies the indices of minimum
    and maximum values in the GAN MSE list.

    Parameters:
    original (list): Original values.
    gan (list): Generated Adversarial Network's values.
    nn (list): Neural Network's values.
    idw (list): Inverse Distance Weighting's values.
    krig (list): Kriging's values.
    natnei (list): Natural Neighbor's values.
    inpt (list): Inpainting values.

    Returns:
    mse_gan_list (list): MSE list for GAN.
    mse_nn_list (list): MSE list for Neural Network.
    mse_idw_list (list): MSE list for IDW.
    mse_krig_list (list): MSE list for Kriging.
    mse_natnei_list (list): MSE list for Natural Neighbor.
    mse_inpt_list (list): MSE list for Inpainting.
    mse_means (list): Mean of each MSE list.
    """

    # Specify the filename
    filename = 'results_comparison_MSE.csv'
    # Join the path and filename
    full_path = os.path.join(path, filename)

    # Initialize MSE lists for each method
    mse_gan_list = []
    mse_nn_list = []
    mse_idw_list = []
    mse_krig_list = []
    mse_natnei_list = []
    mse_inpt_list = []

    # Calculate MSE for each method and append to the respective list
    for i in range(len(original)):
        mse_gan_list.append(np.mean(np.square(gan[i] - original[i])))
        mse_nn_list.append(np.mean(np.square(nn[i] - original[i])))
        mse_idw_list.append(np.mean(np.square(idw[i] - original[i])))
        mse_krig_list.append(np.mean(np.square(krig[i] - original[i])))
        mse_natnei_list.append(np.mean(np.square(natnei[i] - original[i])))
        mse_inpt_list.append(np.mean(np.square(inpt[i] - original[i])))

    # Calculate the mean of each MSE list and store in mse_means
    mse_means = [np.mean(mse_gan_list), np.mean(mse_nn_list),
                 np.mean(mse_idw_list), np.mean(mse_krig_list),
                 np.mean(mse_natnei_list), np.mean(mse_inpt_list)]

    # Save results to a CSV file at the specified path
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'MSE GAN', 'MSE NearNei', 'MSE IDW', 'MSE Krig', 'MSE NatNei', 'MSE Inpt'])
        for i in range(len(original)):
            writer.writerow([i, mse_gan_list[i], mse_nn_list[i],
                             mse_idw_list[i], mse_krig_list[i], mse_natnei_list[i], mse_inpt_list[i]])

        # Add another row in the CSV with the mean of each MSE list
        writer.writerow(['Mean', mse_means[0], mse_means[1], mse_means[2], mse_means[3], mse_means[4], mse_means[5]])

    # Find the index of the minimum and maximum values in mse_gan_list
    min_index = np.argmin(mse_gan_list)
    max_index = np.argmax(mse_gan_list)

    # Print the index of the minimum and maximum values
    print("Index of minimum value in mse_gan_list:", min_index)
    print("Index of maximum value in mse_gan_list:", max_index)

    return mse_gan_list, mse_nn_list, mse_idw_list, mse_krig_list, mse_natnei_list, mse_inpt_list, mse_means




def compute_ssim(original, gan, nn, idw, krig, natnei, inpt, path):
    """
    This function computes the Structural Similarity Index (SSIM) for different algorithms
    and save the results in a CSV file. It also identifies the indices of minimum
    and maximum values in the GAN SSIM list.

    Parameters:
    original (list): Original values.
    gan (list): Generated Adversarial Network's values.
    nn (list): Neural Network's values.
    idw (list): Inverse Distance Weighting's values.
    krig (list): Kriging's values.
    natnei (list): Natural Neighbor's values.
    inpt (list): Inpainting values.

    Returns:
    ssim_gan_list (list): SSIM list for GAN.
    ssim_gan_list_map (list): SSIM map list for GAN.
    ssim_nn_list (list): SSIM list for Neural Network.
    ssim_nn_list_map (list): SSIM map list for Neural Network.
    ssim_idw_list (list): SSIM list for IDW.
    ssim_idw_list_map (list): SSIM map list for IDW.
    ssim_krig_list (list): SSIM list for Kriging.
    ssim_krig_list_map (list): SSIM map list for Kriging.
    ssim_natnei_list (list): SSIM list for Natural Neighbor.
    ssim_natnei_list_map (list): SSIM map list for Natural Neighbor.
    ssim_inpt_list (list): SSIM list for Inpainting.
    ssim_inpt_list_map (list): SSIM map list for Inpainting.
    ssim_means (list): Mean of each SSIM list.
    """
    filename = 'results_comparison_SSIM.csv'
    full_path = os.path.join(path, filename)

    ssim_gan_list, ssim_gan_list_map = [], []
    ssim_nn_list, ssim_nn_list_map = [], []
    ssim_idw_list, ssim_idw_list_map = [], []
    ssim_krig_list, ssim_krig_list_map = [], []
    ssim_natnei_list, ssim_natnei_list_map = [], []
    ssim_inpt_list, ssim_inpt_list_map = [], []

    gan = np.squeeze(gan)
    original = np.squeeze(original)
    nn = np.squeeze(nn)
    idw = np.squeeze(idw)
    krig = np.squeeze(krig)
    natnei = np.squeeze(natnei)
    inpt = np.squeeze(inpt)

    for i in range(len(original)):
        ssim, diff_image = metrics.structural_similarity(gan[i], original[i], data_range=4.5-0, full=True)
        ssim_gan_list.append(ssim)
        ssim_gan_list_map.append(diff_image)

        ssim, diff_image = metrics.structural_similarity(nn[i], original[i], data_range=4.5-0, full=True)
        ssim_nn_list.append(ssim)
        ssim_nn_list_map.append(diff_image)

        ssim, diff_image = metrics.structural_similarity(idw[i], original[i], data_range=4.5-0, full=True)
        ssim_idw_list.append(ssim)
        ssim_idw_list_map.append(diff_image)

        ssim, diff_image = metrics.structural_similarity(krig[i], original[i], data_range=4.5-0, full=True)
        ssim_krig_list.append(ssim)
        ssim_krig_list_map.append(diff_image)

        ssim, diff_image = metrics.structural_similarity(natnei[i], original[i], data_range=4.5-0, full=True)
        ssim_natnei_list.append(ssim)
        ssim_natnei_list_map.append(diff_image)

        ssim, diff_image = metrics.structural_similarity(inpt[i], original[i], data_range=4.5-0, full=True)
        ssim_inpt_list.append(ssim)
        ssim_inpt_list_map.append(diff_image)

    ssim_means = [np.mean(ssim) for ssim in [ssim_gan_list, ssim_nn_list, ssim_idw_list,
                                             ssim_krig_list, ssim_natnei_list, ssim_inpt_list]]

    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'SSIM GAN', 'SSIM NN', 'SSIM IDW',
                         'SSIM Krig', 'SSIM NatNei', 'SSIM Inpt'])
        for i in range(len(original)):
            writer.writerow([i, ssim_gan_list[i], ssim_nn_list[i],
                             ssim_idw_list[i], ssim_krig_list[i],
                             ssim_natnei_list[i], ssim_inpt_list[i]])

        writer.writerow(['Mean', *ssim_means])

    return (ssim_gan_list, ssim_gan_list_map, ssim_nn_list, ssim_nn_list_map,
            ssim_idw_list, ssim_idw_list_map, ssim_krig_list, ssim_krig_list_map,
            ssim_natnei_list, ssim_natnei_list_map, ssim_inpt_list, ssim_inpt_list_map, ssim_means)





def compute_hausdorff_distance(original, gan, nn, idw, krig, natnei, inpt, path):
    """
    This function computes the Hausdorff distance for different algorithms
    and save the results in a CSV file.

    Parameters:
    original (list): Original values.
    gan (list): Generated Adversarial Network's values.
    nn (list): Neural Network's values.
    idw (list): Inverse Distance Weighting's values.
    krig (list): Kriging's values.
    natnei (list): Natural Neighbor's values.
    inpt (list): Inpainting values.

    Returns:
    haus_error_lists (dict): A dictionary containing Hausdorff distance lists for each method.
    """
    filename = 'results_comparison_Hausdorff.csv'
    full_path = os.path.join(path, filename)

    # Initialize lists to store Hausdorff distance for each method
    haus_error_gan_list, haus_error_nn_list, haus_error_idw_list, haus_error_krig_list, haus_error_natnei_list, haus_error_inpt_list = [], [], [], [], [], []

    gan = np.squeeze(gan)
    original = np.squeeze(original)
    nn = np.squeeze(nn)
    idw = np.squeeze(idw)
    krig = np.squeeze(krig)
    natnei = np.squeeze(natnei)
    inpt = np.squeeze(inpt)

    for i in range(len(original)):

        haus_error_gan_list.append(metrics.hausdorff_distance(gan[i], original[i], method='modified'))
        haus_error_nn_list.append(metrics.hausdorff_distance(nn[i], original[i], method='modified'))
        haus_error_idw_list.append(metrics.hausdorff_distance(idw[i], original[i], method='modified'))
        haus_error_krig_list.append(metrics.hausdorff_distance(krig[i], original[i], method='modified'))
        haus_error_natnei_list.append(metrics.hausdorff_distance(natnei[i], original[i], method='modified'))
        haus_error_inpt_list.append(metrics.hausdorff_distance(inpt[i], original[i], method='modified'))

    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Haus GAN', 'Haus NN', 'Haus IDW',
                         'Haus Krig', 'Haus NatNei', 'Haus Inpt'])
        for i in range(len(original)):
            writer.writerow([i, haus_error_gan_list[i], haus_error_nn_list[i],
                             haus_error_idw_list[i], haus_error_krig_list[i],
                             haus_error_natnei_list[i], haus_error_inpt_list[i]])


    return haus_error_gan_list, haus_error_nn_list, haus_error_idw_list, haus_error_krig_list, haus_error_natnei_list, haus_error_inpt_list





def compute_errors_gan_only(original, gan, path):
    """
    This function computes the Mean Absolute Errors (MAE) for GAN method
    and save the results in a CSV file.

    Parameters:
    original (list): Original values.
    gan (list): Generated Adversarial Network's values.

    Returns:
    mae_gan (list): MAE list for GAN.
    mae_gan_avg (float): Average MAE for GAN.
    mae_gan_stddev (float): Standard deviation of MAE for GAN.
    """

    # Specify the filename
    filename = 'results_comparison_MAE_gan_only.csv'
    # Join the path and filename
    full_path = os.path.join(path, filename)

    # Initialize MAE list for GAN
    mae_gan = []

    # Calculate MAE for GAN and append to the list
    for i in range(len(original)):
        mae_gan.append(np.mean(np.abs(gan[i] - original[i])))

    # Calculate the average and standard deviation of the MAE list
    mae_gan_avg = np.mean(mae_gan)
    mae_gan_stddev = np.std(mae_gan)

    # Save results to a CSV file at the specified path
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'MAE GAN'])
        for i in range(len(original)):
            writer.writerow([i, mae_gan[i]])

        # Add another row in the CSV with the mean and stddev of the MAE list
        writer.writerow(['Average', mae_gan_avg])
        writer.writerow(['Standard Deviation', mae_gan_stddev])

    # Find the index of the minimum and maximum values in mae_gan_list
    min_index = np.argmin(mae_gan)
    max_index = np.argmax(mae_gan)

    # Print the index of the minimum and maximum values
    print("Index of minimum value in mae_gan_list:", min_index)
    print("Index of maximum value in mae_gan_list:", max_index)

    return mae_gan, mae_gan_avg, mae_gan_stddev







