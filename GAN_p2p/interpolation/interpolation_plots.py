import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set the font family to "Arial"
rcParams['font.family'] = 'Arial'

def plot_histograms_row(gan, NeNI, idw, krig, NatNI, inpt):
    """
    Plots three histograms in a row for GAN, Nearest Neighbor, IDW and Kriging.

    Parameters:
    gan, nn, idw, krig (np.array): Arrays containing the Mean Absolute Error (MAE)
    for GAN, Nearest Neighbor, IDW, and Kriging methods respectively.
    """

    # Create a figure with 5 subplots in a row
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    # Plot histogram for GAN vs Nearest Neighbor in first subplot
    axs[0].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[0].hist(NeNI, bins=20, alpha=0.5, label='NeNI', color='goldenrod', edgecolor='dimgray')
    axs[0].axvline(np.mean(gan), color='dimgrey', linestyle='dashed', linewidth=1)
    axs[0].axvline(np.mean(NeNI), color='darkgoldenrod', linestyle='dashed', linewidth=1)
    axs[0].set_xlabel('Mean absolute error', fontsize=10)
    axs[0].set_ylabel('Frequency', fontsize=10)
    axs[0].set_title('Mean absolute error GAN vs Nearest Neighbor', fontsize=10)
    axs[0].legend(loc='upper right', fontsize=7)
    axs[0].set_xlim(0.0, 0.5)
    axs[0].tick_params(axis='both', labelsize=8)

    # Plot histogram for GAN vs IDW in second subplot
    axs[1].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[1].hist(idw, bins=40, alpha=0.5, label='IDW', color='teal', edgecolor='dimgray')
    axs[1].axvline(np.mean(gan), color='dimgrey', linestyle='dashed', linewidth=1)
    axs[1].axvline(np.mean(idw), color='teal', linestyle='dashed', linewidth=1)
    axs[1].set_xlabel('Mean absolute error', fontsize=10)
    axs[1].set_ylabel('Frequency', fontsize=10)
    axs[1].set_title('Mean absolute error for GAN vs IDW', fontsize=10)
    axs[1].legend(loc='upper right', fontsize=7)
    axs[1].set_xlim(0.0, 0.5)
    axs[1].tick_params(axis='both', labelsize=8)

    # Plot histogram for GAN vs Kriging in third subplot
    axs[2].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[2].hist(krig, bins=20, alpha=0.5, label='Krig', color='indigo', edgecolor='dimgray')
    axs[2].axvline(np.mean(gan), color='dimgrey', linestyle='dashed', linewidth=1)
    axs[2].axvline(np.mean(krig), color='indigo', linestyle='dashed', linewidth=1)
    axs[2].set_xlabel('Mean absolute error', fontsize=10)
    axs[2].set_ylabel('Frequency', fontsize=10)
    axs[2].set_title('Mean absolute error for GAN vs Kriging', fontsize=10)
    axs[2].legend(loc='upper right', fontsize=7)
    axs[2].set_xlim(0.0, 0.5)
    axs[2].tick_params(axis='both', labelsize=8)


    # Plot histogram for GAN vs NatNei in fourth subplot
    axs[3].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[3].hist(NatNI, bins=20, alpha=0.5, label='NatNI', color='darkgreen', edgecolor='dimgray')
    axs[3].axvline(np.mean(gan), color='dimgrey', linestyle='dashed', linewidth=1)
    axs[3].axvline(np.mean(NatNI), color='darkgreen', linestyle='dashed', linewidth=1)
    axs[3].set_xlabel('Mean absolute error', fontsize=10)
    axs[3].set_ylabel('Frequency', fontsize=10)
    axs[3].set_title('Mean absolute error for GAN vs NatNei', fontsize=10)
    axs[3].legend(loc='upper right', fontsize=7)
    axs[3].set_xlim(0.0, 0.5)
    axs[3].tick_params(axis='both', labelsize=8)


    # Plot histogram for GAN vs Inpt in fifth subplot
    axs[4].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[4].hist(inpt, bins=20, alpha=0.5, label='Inpt', color='maroon', edgecolor='dimgray')
    axs[4].axvline(np.mean(gan), color='dimgrey', linestyle='dashed', linewidth=1)
    axs[4].axvline(np.mean(inpt), color='maroon', linestyle='dashed', linewidth=1)
    axs[4].set_xlabel('Mean absolute error', fontsize=10)
    axs[4].set_ylabel('Frequency', fontsize=10)
    axs[4].set_title('Mean absolute error for GAN vs Inpt', fontsize=10)
    axs[4].legend(loc='upper right', fontsize=7)
    axs[4].set_xlim(0.0, 0.5)
    axs[4].tick_params(axis='both', labelsize=8)


    # Adjust the spacing between subplots
    plt.tight_layout()

def plot_comparison_of_methods(src_img, gen_img, tar_img, nn, idw, kriging, natni, inp, mae_means):
    """
    This function plots the comparison of different methods' results including GAN,
    nearest neighbor interpolation, inverse distance interpolation, kriging interpolation,
    Natural Neighbor interpolation, and Inpainting.

    Parameters:
    src_img, gen_img, tar_img, nn, idw, kriging, natni, inp (np.array): Arrays containing source image,
    generated image, target image, nearest neighbor interpolated image,
    inverse distance interpolated image, kriging interpolated image,
    Natural Neighbor interpolated image, and Inpainting image.
    mae_means (np.array): Mean absolute error of the methods
    """

    # Stack all the images
    images = np.vstack((tar_img, src_img,
                        gen_img, np.abs(gen_img - tar_img),
                        nn, np.abs(nn - tar_img),
                        idw, np.abs(idw - tar_img),
                        kriging, np.abs(kriging - tar_img),
                        natni, np.abs(natni - tar_img),
                        inp, np.abs(inp - tar_img)))

    # Set plot titles for each subplot
    titles = ['Original cross-section', 'Input CPT data',
              'SchemaGAN prediction', f'SchemaGAN MAE: {mae_means[0]:.2f}',
              'Nearest Neighbor', f'Nearest Neighbor MAE: {mae_means[1]:.2f}',
              'Inverse Distance Weight', f'Inverse Distance MAE: {mae_means[2]:.2f}',
              'Ordinary Kriging', f'Ordinary Kriging MAE: {mae_means[3]:.2f}',
              'Natural Neighbor', f'Natural Neighbor MAE: {mae_means[4]:.2f}',
              'Inpainting', f'Inpainting MAE: {mae_means[5]:.2f}']

    # Set the axis labels for each subplot
    xlabels = ['Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Distance', 'Distance']
    ylabels = ['Depth', '', 'Depth', '', 'Depth', '', 'Depth', '', 'Depth', '', 'Depth', '', 'Depth', '']

    # Set the colorbar range for each subplot
    ranges_vmin_vmax = [[1, 4.5], [0, 4.5],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1]]

    # Set the colorbar titles for each subplot
    cbar_titles = ['', '', '', '', '', '', '', '', '', '', '', '', 'Ic values', 'Ic error values']

    # Follow the rest of your plotting code accordingly

    # Define the number of subplots for the overall figure
    num_images = len(images)
    num_rows = 7
    num_cols = int(np.ceil(num_images / num_rows))

    # Create a figure
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    # Plot images row by row
    for i in range(num_images):
        # Define subplot
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]

        # Use 'viridis' cmap for images on the left (even column index), and 'inferno' for images on the right (odd column index)
        cmap = 'viridis' if col % 2 == 0 else 'viridis'

        im = ax.imshow(images[i, :, :, 0], cmap=cmap, vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])

        # Set title and font size
        ax.set_title(titles[i], fontsize=9)

        # Set tick_params and font size
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='both', which='minor', labelsize=9)

        # Set x and y labels and font size
        ax.set_xlabel(xlabels[i], fontsize=9)
        ax.set_ylabel(ylabels[i], fontsize=9)

        # Manually set tick mark spacing
        ax.set_xticks(np.arange(0, images.shape[2], 50))
        ax.set_yticks(np.arange(0, images.shape[1], 15))

        # Set the aspect ratio of the subplot
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([0, 511])
        ax.set_ylim([31, 0])

        # Add colorbar to all subplots
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.08, pad=0.5, aspect=40)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(cbar_titles[i], fontsize=8)
        ax.set_xlim([0, 511])
        ax.set_ylim([31, 0])

    # Adjust the spacing between subplots
    plt.tight_layout()





def generate_boxplot(gan, nearnei, idw, krig, natnei, inpt):
    """
    Generate a box plot to compare the Mean Absolute Error (MAE) values
    of different interpolation methods.

    Parameters:
        gan (list): List of MAE values for SchemaGAN method.
        nearnei (list): List of MAE values for Nearest Neighbor method.
        idw (list): List of MAE values for IDW method.
        krig (list): List of MAE values for Kriging method.
        natnei (list): List of MAE values for Natural Neighbor method.
        inpt (list): List of MAE values for Inpainting method.

    Returns:
        None
    """

    # Combine the MAE values into a list of lists
    data = [gan, nearnei, idw, krig, natnei, inpt]

    # Create a figure and axis object with a specific size
    fig, ax = plt.subplots(figsize=(9, 3))

    # Define the color palette
    color_palette = plt.cm.viridis

    # Create the box plot with filled boxes
    boxplot = ax.boxplot(data, showfliers=True, sym='.', whis=[5, 95], patch_artist=True)

    # Customize the colors of the boxes
    for i, box in enumerate(boxplot['boxes']):
        # Set the facecolor of the boxes using the Viridis palette
        box.set(facecolor=color_palette(i / len(data)), alpha=0.5)

    # Change the color of the mean line inside the boxes to black
    for median in boxplot['medians']:
        # Set the color of the median line to black
        median.set(color='black')

    # Customize the plot labels
    ax.set_xticklabels(['SchemaGAN', 'Nearest Neighbour', 'Kriging', 'IDW', 'Natural Neighbour', 'Inpainting'])

    # Set the y-axis label
    ax.set_ylabel('MAE')

    # Set the plot title
    ax.set_title('Comparison of Interpolation Methods')

    # Show the plot
    plt.show()