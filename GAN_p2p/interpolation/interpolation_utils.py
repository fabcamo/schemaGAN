import numpy as np
from tensorflow.keras.models import load_model
from GAN_p2p.functions.p2p_process_data import reverse_IC_normalization
from methods import nearest_interpolation



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











