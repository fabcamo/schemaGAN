import numpy as np
import csv
from tensorflow.keras.models import load_model
from GAN_p2p.functions.p2p_process_data import reverse_IC_normalization
from methods import nearest_interpolation, idw_interpolation, kriging_interpolation, natural_nei_interpolation



# Grab the data from the cpt-like data image (src_image)
def get_cptlike_data(src_images):

    coords_all = []  # to store the coordinates
    pixel_values_all = []  # to store the pixel values

    # Loop over each image in src_images to grab the coordinates with IC values
    for i in range(src_images.shape[0]):
        # Get the indices of non-zero values in the i-th image
        # y_indices>[rows] & x_indices>[cols]
        y_indices, x_indices = np.nonzero(src_images[i, :, :, 0])
        # Combine the x and y indices into a 2D array
        # in format> (rows, cols)
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


# Format the input data for the plots
def format_source_images(dataset):
    # call the dataset (normalized)
    [input_img, orig_img] = dataset

    original_img = []
    cptlike_img = []

    for i in range(input_img.shape[0]):
        # Choose a cross-section to run through the generator
        cross_section_number = i
        # Choose a given cross-seciton
        ix = np.array([cross_section_number])
        # call the {i} cpt-like image and the original image
        src_image, tar_image = input_img[ix], orig_img[ix]

        # Reverse normalize the original and cpt-like images> scale from [-1,1] to [0,255]
        src_image = reverse_IC_normalization(src_image)
        tar_image = reverse_IC_normalization(tar_image)

        cptlike_img.append(src_image)
        original_img.append(tar_image)

    return original_img, cptlike_img




def generate_gan_image(generator_path, dataset):
    # call the dataset (normalized)
    [input_img, orig_img] = dataset

    # Load the generator model from path
    model = load_model(generator_path)

    gan_images = []
    for i in range(input_img.shape[0]):
        # Choose a cross-section to run through the generator
        cross_section_number = i
        # Choose a given cross-seciton
        ix = np.array([cross_section_number])
        # call the {i} cpt-like image and the original image
        src_image, tar_image = input_img[ix], orig_img[ix]

        gan_generated_image = model.predict(src_image)
        # Reverse normalize the original and cpt-like images> scale from [-1,1] to [0,255]
        gan_generated_image = reverse_IC_normalization(gan_generated_image)
        gan_images.append(gan_generated_image)

    return gan_images


def generate_nn_images(no_rows, no_cols, src_images):
    # Create 2D grid with specified number of rows and columns
    rows = np.linspace(0, no_rows - 1, no_rows)
    cols = np.linspace(0, no_cols - 1, no_cols)
    grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)

    coords_all, pixel_values_all = get_cptlike_data(src_images)

    nn_images = []
    for i in range(src_images.shape[0]):

        # call the {i} coordinates with pixels and pixel values
        coords, pixel_values = coords_all[i], pixel_values_all[i]

        nn_interpolation = nearest_interpolation(coords, pixel_values, grid)
        # Reshape the results of a single image to plot
        nn_interpolation = np.reshape(nn_interpolation, (1, no_rows, no_cols, 1))
        # Append the results to the list
        nn_images.append(nn_interpolation)

    return nn_images


def generate_idw_images(no_rows, no_cols, src_images):
    # Create 2D grid with specified number of rows and columns
    rows = np.linspace(0, no_rows - 1, no_rows)
    cols = np.linspace(0, no_cols - 1, no_cols)
    grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)

    coords_all, pixel_values_all = get_cptlike_data(src_images)

    idw_images = []
    for i in range(src_images.shape[0]):

        # call the {i} coordinates with pixels and pixel values
        coords, pixel_values = coords_all[i], pixel_values_all[i]

        idw_inter = idw_interpolation(coords, pixel_values, grid)
        # Reshape the results of a single image to plot
        idw_inter = np.reshape(idw_inter, (1, no_rows, no_cols, 1))
        # Append the results to the list
        idw_images.append(idw_inter)

    return idw_images


def generate_krig_images(no_rows, no_cols, src_images):
    # Create 2D grid with specified number of rows and columns
    gridx = np.linspace(0, no_cols - 1, no_cols)
    gridy = np.linspace(0, no_rows - 1, no_rows)

    coords_all, pixel_values_all = get_cptlike_data(src_images)

    krig_images = []
    for i in range(src_images.shape[0]):

        # call the {i} coordinates with pixels and pixel values
        coords, pixel_values = coords_all[i], pixel_values_all[i]

        # Run the interpolation
        krig_inter = kriging_interpolation(coords, pixel_values, gridx, gridy)
        # Reshape the results of a single image to plot
        krig_inter = np.reshape(krig_inter, (1, no_rows, no_cols, 1))

        # Append the results to the list
        krig_images.append(krig_inter)

    return krig_images



def generate_nat_nei_images(no_rows, no_cols, src_images):
    # Create 2D grid with specified number of rows and columns
    rows = np.linspace(0, no_rows - 1, no_rows)
    cols = np.linspace(0, no_cols - 1, no_cols)
    grid = np.array(np.meshgrid(rows, cols)).T.reshape(-1, 2)

    coords_all, pixel_values_all = get_cptlike_data(src_images)

    natnei_images = []
    for i in range(src_images.shape[0]):

        # call the {i} coordinates with pixels and pixel values
        coords, pixel_values = coords_all[i], pixel_values_all[i]

        natnei_inter = natural_nei_interpolation(coords, pixel_values, grid)
        # Reshape the results of a single image to plot
        natnei_inter = np.reshape(natnei_inter, (1, no_rows, no_cols, 1))
        # Append the results to the list
        natnei_images.append(natnei_inter)

    return natnei_images





def compute_errors(original, gan, nn, idw, krig, natnei):

    mae_gan_list = []
    mae_nn_list = []
    mae_idw_list = []
    mae_krig_list = []
    mae_natnei_list = []

    for i in range(len(original)):
        mae_gan = np.mean(np.abs(gan[i] - original[i]))
        mae_gan_list.append(mae_gan)

        mae_nn = np.mean(np.abs(nn[i] - original[i]))
        mae_nn_list.append(mae_nn)

        mae_idw = np.mean(np.abs(idw[i] - original[i]))
        mae_idw_list.append(mae_idw)

        mae_krig = np.mean(np.abs(krig[i] - original[i]))
        mae_krig_list.append(mae_krig)

        mae_natnei = np.mean(np.abs(natnei[i] - original[i]))
        mae_natnei_list.append(mae_natnei)

    # Save results to a CSV file
    with open('results_comparison_MAE.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'MAE GAN', 'MAE NN', 'MAE IDW', 'MAE Krig', 'MAE NatNei'])
        for i in range(len(original)):
            writer.writerow(
                [i, mae_gan_list[i], mae_nn_list[i], mae_idw_list[i], mae_krig_list[i], mae_natnei_list[i]])

    # Find the index of the minimum and maximum values in mae_gan_list
    min_index = np.argmin(mae_gan_list)
    max_index = np.argmax(mae_gan_list)
    # Print the index of the minimum and maximum values
    print("Index of minimum value in mae_gan_list:", min_index)
    print("Index of maximum value in mae_gan_list:", max_index)

    return mae_gan_list, mae_nn_list, mae_idw_list, mae_krig_list, mae_natnei_list








