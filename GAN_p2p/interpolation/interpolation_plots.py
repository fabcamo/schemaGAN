import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_images_error_comparison(src_img, generated_img, tar_img):

    # Calculate the Mean absolute error between the target image and the generated one
    mae = np.mean(np.abs(tar_img - generated_img))

    # Stack all the images
    images = np.vstack((src_img, generated_img, tar_img, np.abs(generated_img-tar_img)))

    # Set plot titles
    titles = ['Input', 'Output-Generated', 'Original', f'Mean absolute error: {mae:.2f}']
    # Set the cbar range
    ranges_vmin_vmax = [[0, 4.5], [0, 4.5], [0, 4.5], [0, 1]]
    # Set the cbar titles
    cbar_titles = ['IC', 'IC', 'IC', 'IC error']

    # Create a figure with a size of 10 inches by 4 inches
    fig = plt.figure(figsize=(10, 15))
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        ax = fig.add_subplot(4, 1, 1 + i)
        im = ax.imshow(images[i, :, :,0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])
        # set title with fontsize
        ax.set_title(titles[i], fontsize=10)
        # set tick_params with fontsize
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='both', which='minor', labelsize=9)
        # set x and y labels
        ax.set_xlabel('Distance', fontsize=9)
        ax.set_ylabel('Depth', fontsize=9)
        # manually set tick mark spacing
        ax.set_xticks(np.arange(0, images.shape[2], 40))
        ax.set_yticks(np.arange(0, images.shape[1], 20))
        # add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.16, shrink=0.7)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(cbar_titles[i], fontsize=9)
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=6)
        cbar.update_ticks()



def plot_histograms(list1, list2, name):
    # create a figure and axis object
    fig, ax = plt.subplots()

    # plot histograms of the two lists on the same axis with outlines of the bars
    ax.hist(list1, alpha=0.5, label='GAN', bins=10, color='gray', edgecolor='dimgray')
    ax.hist(list2, alpha=0.5, label='NN', bins=20, color='goldenrod', edgecolor='dimgray')

    # calculate means of the two lists
    mean1 = np.mean(list1)
    mean2 = np.mean(list2)

    # plot vertical lines at the means of the two lists
    ax.axvline(mean1, color='dimgrey', linestyle='dashed', linewidth=1)
    ax.axvline(mean2, color='darkgoldenrod', linestyle='dashed', linewidth=1)

    # add labels and legend
    plt.xlabel('Mean absolute error')
    plt.ylabel('Frequency')
    plt.title(f'Mean absolute error for {name}')
    ax.legend(loc='upper right')

    # display the plot
    plt.show()


def plot_histograms_row(gan, nn, idw, krig):
    # create a figure with 3 subplots in a row
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    bins0 = np.histogram(np.hstack((gan, nn)), bins=20)[1]  # get the bin edges
    bins1 = np.histogram(np.hstack((gan, idw)), bins=20)[1]  # get the bin edges
    bins2 = np.histogram(np.hstack((gan, krig)), bins=20)[1]  # get the bin edges

    # plot each histogram in a separate subplot
    axs[0].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[0].hist(nn, bins=20, alpha=0.5, label='NN', color='goldenrod', edgecolor='dimgray')
    mean1a = np.mean(gan)
    mean2a = np.mean(nn)
    axs[0].axvline(mean1a, color='dimgrey', linestyle='dashed', linewidth=1)
    axs[0].axvline(mean2a, color='darkgoldenrod', linestyle='dashed', linewidth=1)
    axs[0].set_xlabel('Mean absolute error')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Mean absolute error GAN vs NN')
    axs[0].legend(loc='upper right')
    axs[0].set_xlim(0.05, 0.5)

    axs[1].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[1].hist(idw, bins=20, alpha=0.5, label='IDW', color='teal', edgecolor='dimgray')
    mean1b = np.mean(gan)
    mean2b = np.mean(idw)
    axs[1].axvline(mean1b, color='dimgrey', linestyle='dashed', linewidth=1)
    axs[1].axvline(mean2b, color='teal', linestyle='dashed', linewidth=1)
    axs[1].set_xlabel('Mean absolute error')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Mean absolute error for GAN vs IDW')
    axs[1].legend(loc='upper right')
    axs[1].set_xlim(0.05, 0.5)

    axs[2].hist(gan, bins=10, alpha=0.5, label='GAN', color='gray', edgecolor='dimgray')
    axs[2].hist(krig, bins=20, alpha=0.5, label='Krig', color='indigo', edgecolor='dimgray')
    mean1c = np.mean(gan)
    mean2c = np.mean(krig)
    axs[2].axvline(mean1c, color='dimgrey', linestyle='dashed', linewidth=1)
    axs[2].axvline(mean2c, color='indigo', linestyle='dashed', linewidth=1)
    axs[2].set_xlabel('Mean absolute error')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title(f'Mean absolute error for GAN vs Kriging')
    axs[2].legend(loc='upper right')
    axs[2].set_xlim(0.05, 0.5)

    # adjust the spacing between subplots
    plt.tight_layout()

    plt.savefig('histograms.png')
    # display the plot
    plt.show()


def plot_comparison_of_methods(src_img, gen_img, tar_img, nn, idw, kriging):

    # Stack all the images
    images = np.vstack((tar_img, src_img,
                        gen_img, np.abs(gen_img-tar_img),
                        nn, np.abs(nn-tar_img),
                        idw, np.abs(idw-tar_img),
                        kriging, np.abs(kriging-tar_img)))

    # Set plot titles for each subplot
    titles = ['Original cross-section', 'Input CPT data',
              'GAN prediction', 'GAN MAE:',
              'Nearest neighbor interpolation', 'Nearest neighbor MAE:',
              'Inverse distance interpolation', 'Inverse distance MAE:',
              'Ordinary kriging interpolation', 'Ordinary kriging MAE:']

    # Set the axis labels for each subplot
    xlabels = ['', '', '', '', '', '', '', '', 'Distance', 'Distance']
    ylabels = ['Depth', '', 'Depth', '', 'Depth', '', 'Depth', '', 'Depth', '']

    # Set the cbar range for each subplot
    ranges_vmin_vmax = [[1, 4.5], [0, 4.5],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1],
                        [1, 4.5], [0, 1]]

    # Set the cbar titles for each subplot
    cbar_titles = ['', '', '', '', '', '', '', '', 'IC values', 'IC error values']

    # Defining the number of subplots to use in the overall figure
    num_images = len(images)
    num_rows = 5
    num_cols = int(np.ceil(num_images / num_rows))

    # Create a figure with a size of 8 inches by 5 inches
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 5))

    # Plot images row by row
    for i in range(num_images):
        # Define subplot
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        im = ax.imshow(images[i, :, :, 0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])

        # Set title with fontsize
        ax.set_title(titles[i], fontsize=8)

        # Set tick_params with fontsize
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=7)

        # Set x and y labels
        ax.set_xlabel(xlabels[i], fontsize=7)
        ax.set_ylabel(ylabels[i], fontsize=7)

        # Manually set tick mark spacing
        ax.set_xticks(np.arange(0, images.shape[2], 50))
        ax.set_yticks(np.arange(0, images.shape[1], 15))

        # Set the size of the subplot
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

    plt.tight_layout()




