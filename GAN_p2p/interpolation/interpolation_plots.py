import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def plot_histograms_row(gan, nn, idw, krig):
    """
    Plots three histograms in a row for GAN, Nearest Neighbor, IDW and Kriging.

    Parameters:
    gan, nn, idw, krig (np.array): Arrays containing the Mean Absolute Error (MAE)
    for GAN, Nearest Neighbor, IDW, and Kriging methods respectively.
    """

    # Create a figure with 3 subplots in a row
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Get the bin edges for histograms
    bins0 = np.histogram(np.hstack((gan, nn)), bins=20)[1]
    bins1 = np.histogram(np.hstack((gan, idw)), bins=20)[1]
    bins2 = np.histogram(np.hstack((gan, krig)), bins=20)[1]

    # Plot histogram for GAN vs Nearest Neighbor in first subplot
    axs[0].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[0].hist(nn, bins=20, alpha=0.5, label='NearNei', color='goldenrod', edgecolor='dimgray')
    axs[0].axvline(np.mean(gan), color='dimgrey', linestyle='dashed', linewidth=1)
    axs[0].axvline(np.mean(nn), color='darkgoldenrod', linestyle='dashed', linewidth=1)
    axs[0].set_xlabel('Mean absolute error')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Mean absolute error GAN vs Nearest Neighbor')
    axs[0].legend(loc='upper right')
    axs[0].set_xlim(0.05, 0.5)

    # Plot histogram for GAN vs IDW in second subplot
    axs[1].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[1].hist(idw, bins=20, alpha=0.5, label='IDW', color='teal', edgecolor='dimgray')
    axs[1].axvline(np.mean(gan), color='dimgrey', linestyle='dashed', linewidth=1)
    axs[1].axvline(np.mean(idw), color='teal', linestyle='dashed', linewidth=1)
    axs[1].set_xlabel('Mean absolute error')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Mean absolute error for GAN vs IDW')
    axs[1].legend(loc='upper right')
    axs[1].set_xlim(0.05, 0.5)

    # Plot histogram for GAN vs Kriging in third subplot
    axs[2].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[2].hist(krig, bins=20, alpha=0.5, label='Krig', color='indigo', edgecolor='dimgray')
    axs[2].axvline(np.mean(gan), color='dimgrey', linestyle='dashed', linewidth=1)
    axs[2].axvline(np.mean(krig), color='indigo', linestyle='dashed', linewidth=1)
    axs[2].set_xlabel('Mean absolute error')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('Mean absolute error for GAN vs Kriging')
    axs[2].legend(loc='upper right')
    axs[2].set_xlim(0.05, 0.5)

    # Adjust the spacing between subplots
    plt.tight_layout()




def plot_comparison_of_methods(src_img, gen_img, tar_img, nn, idw, kriging, mae_means):
    """
    This function plots the comparison of different methods' results including GAN,
    nearest neighbor interpolation, inverse distance interpolation, and kriging interpolation.

    Parameters:
    src_img, gen_img, tar_img, nn, idw, kriging (np.array): Arrays containing source image,
    generated image, target image, nearest neighbor interpolated image,
    inverse distance interpolated image, and kriging interpolated image.
    mae_means (np.array): Mean absolute error of the methods
    """

    # Stack all the images
    images = np.vstack((tar_img, src_img,
                        gen_img, np.abs(gen_img - tar_img),
                        nn, np.abs(nn - tar_img),
                        idw, np.abs(idw - tar_img),
                        kriging, np.abs(kriging - tar_img)))

    # Set plot titles for each subplot
    titles = ['Original cross-section', 'Input CPT data',
              'GAN prediction', f'GAN MAE: {mae_means[0]:.2f}',
              'Nearest neighbor interpolation', f'Nearest neighbor MAE: {mae_means[1]:.2f}',
              'Inverse distance interpolation', f'Inverse distance MAE: {mae_means[2]:.2f}',
              'Ordinary kriging interpolation', f'Ordinary kriging MAE: {mae_means[3]:.2f}']

    # Set the axis labels for each subplot
    xlabels = ['', '', '', '', '', '', '', '', 'Distance', 'Distance']
    ylabels = ['Depth', '', 'Depth', '', 'Depth', '', 'Depth', '', 'Depth', '']

    # Set the colorbar range for each subplot
    ranges_vmin_vmax = [[1, 4.5], [0, 4.5],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1]]

    # Set the colorbar titles for each subplot
    cbar_titles = ['', '', '', '', '', '', '', '', 'IC values', 'IC error values']

    # Define the number of subplots for the overall figure
    num_images = len(images)
    num_rows = 5
    num_cols = int(np.ceil(num_images / num_rows))

    # Create a figure
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 5))

    # Plot images row by row
    for i in range(num_images):
        # Define subplot
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        im = ax.imshow(images[i, :, :, 0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])

        # Set title and font size
        ax.set_title(titles[i], fontsize=8)

        # Set tick_params and font size
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=7)

        # Set x and y labels and font size
        ax.set_xlabel(xlabels[i], fontsize=7)
        ax.set_ylabel(ylabels[i], fontsize=7)

        # Manually set tick mark spacing
        ax.set_xticks(np.arange(0, images.shape[2], 50))
        ax.set_yticks(np.arange(0, images.shape[1], 15))

        # Set the aspect ratio of the subplot
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([0, 512])
        ax.set_ylim([32, 0])

        # Add colorbar only to last row
        if row == num_rows - 1:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.08, pad=0.5, aspect=40)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(cbar_titles[i], fontsize=7)
            ax.set_xlim([0, 512])
            ax.set_ylim([32, 0])

    # Adjust the spacing between subplots
    plt.tight_layout()

