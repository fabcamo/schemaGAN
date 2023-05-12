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




