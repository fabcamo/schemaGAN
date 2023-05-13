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
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # plot each histogram in a separate subplot
    axs[0].hist(gan, alpha=0.5, label='GAN', bins=10, color='gray', edgecolor='dimgray')
    axs[0].hist(nn, alpha=0.5, label='NN', bins=20, color='goldenrod', edgecolor='dimgray')
    mean1a = np.mean(gan)
    mean2a = np.mean(nn)
    axs[0].axvline(mean1a, color='dimgrey', linestyle='dashed', linewidth=1)
    axs[0].axvline(mean2a, color='darkgoldenrod', linestyle='dashed', linewidth=1)
    axs[0].set_xlabel('Mean absolute error')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Mean absolute error GAN vs NN')
    axs[0].legend(loc='upper right')

    axs[1].hist(gan, alpha=0.5, label='GAN', bins=10, color='gray', edgecolor='dimgray')
    axs[1].hist(idw, alpha=0.5, label='IDW', bins=20, color='teal', edgecolor='dimgray')
    mean1b = np.mean(gan)
    mean2b = np.mean(idw)
    axs[1].axvline(mean1b, color='dimgrey', linestyle='dashed', linewidth=1)
    axs[1].axvline(mean2b, color='teal', linestyle='dashed', linewidth=1)
    axs[1].set_xlabel('Mean absolute error')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Mean absolute error for GAN vs IDW')
    axs[1].legend(loc='upper right')

    axs[2].hist(gan, alpha=0.5, label='GAN', bins=10, color='gray', edgecolor='dimgray')
    axs[2].hist(krig, alpha=0.5, label='Krig', bins=20, color='indigo', edgecolor='dimgray')
    mean1c = np.mean(gan)
    mean2c = np.mean(krig)
    axs[2].axvline(mean1c, color='dimgrey', linestyle='dashed', linewidth=1)
    axs[2].axvline(mean2c, color='indigo', linestyle='dashed', linewidth=1)
    axs[2].set_xlabel('Mean absolute error')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title(f'Mean absolute error for GAN vs Kriging')
    axs[2].legend(loc='upper right')

    # adjust the spacing between subplots
    plt.tight_layout()

    # display the plot
    plt.show()

