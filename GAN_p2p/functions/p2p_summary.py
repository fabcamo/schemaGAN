import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functions.p2p_generate_samples import generate_real_samples, generate_fake_samples
from functions.p2p_process_data import reverse_normalization



# Save the generator model and check how good the generated image looks.
def summarize_performance(step, g_model, dataset, n_samples=5):
    print('... Saving a summary')

    # Select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)

    # Generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    # Scale all pixels from [-1,1] to [0,1]
    X_realA = reverse_normalization(X_realA)
    X_realB = reverse_normalization(X_realB)
    X_fakeB = reverse_normalization(X_fakeB)

    # plot real source images
    for i in range(n_samples):
        ax = plt.subplot(5, n_samples, 1 + i)
        # plt.axis('off')
        plt.imshow(X_realA[i])
        if i == 0:
            plt.ylabel('Real Source', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xticklabels(['0', '64', '128', '192', '256'], fontsize=6)
        ax.set_yticks([0, 32, 64])
        ax.set_yticklabels(['0', '32', '64'], fontsize=6)

    # plot generated target image
    for i in range(n_samples):
        ax = plt.subplot(5, n_samples, 1 + n_samples + i)
        # plt.axis('off')
        plt.imshow(X_fakeB[i])
        if i == 0:
            plt.ylabel('Generated Target', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xticklabels(['0', '64', '128', '192', '256'], fontsize=6)
        ax.set_yticks([0, 32, 64])
        ax.set_yticklabels(['0', '32', '64'], fontsize=6)

    # plot real target image
    for i in range(n_samples):
        ax = plt.subplot(5, n_samples, 1 + n_samples * 2 + i)
        # plt.axis('off')
        plt.imshow(X_realB[i])
        if i == 0:
            plt.ylabel('Real Target', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xticklabels(['0', '64', '128', '192', '256'], fontsize=6)
        ax.set_yticks([0, 32, 64])
        ax.set_yticklabels(['0', '32', '64'], fontsize=6)

    # plot all three images> Input, generated and original

    filename1 = 'plot_%06d.png' % (step + 1)
    plt.savefig(filename1)
    plt.close()

    [src_image, tar_image], _ = generate_real_samples(dataset, 1, 1)
    gen_image, _ = generate_fake_samples(g_model, src_image, 1)
    plot_images_with_error(src_image, gen_image, tar_image)

    # save the generator model
    filename2 = 'model_%06d.h5' % (step + 1)
    #g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))



# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='d-real')
    plt.plot(d2_hist, label='d-fake')
    plt.plot(d_hist, label='d-total')
    plt.plot(g_hist, label='gen')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plot discriminator accuracy
    plt.subplot(2, 1, 2)
    plt.plot(a1_hist, label='acc-real')
    plt.plot(a2_hist, label='acc-fake')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Save plot to file
    plot_losses = 'plot_losses.png'

    plt.savefig(plot_losses, bbox_inches='tight')
    plt.close()



def plot_images(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img))

    # Scale from [-1,1] to [0,255]
    images = reverse_normalization(images)

    # Set plot titles
    titles = ['Input', 'Output-Generated', 'Original']
    ranges_vmin_vmax = [[1.3, 4.2], [1.3, 4.2], [1.3, 4.2]]

    # Create a figure with a size of 10 inches by 4 inches
    fig = plt.figure(figsize=(15, 5))

    # plot images row by row
    for i in range(len(images)):
        # define subplot
        ax = fig.add_subplot(1, 3, 1 + i)
        # plot raw pixel data
        im = ax.imshow(images[i, :, :, 0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])
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
        cbar.set_label('IC', fontsize=9)
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=6)
        cbar.update_ticks()


    plt.show()





def plot_images_error(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img, np.abs(gen_img-tar_img)))
    test = np.abs(gen_img-tar_img)
    print(np.min(test))
    print(np.max(test))

    # Scale from [-1,1] to [0,255]
    images = reverse_normalization(images)

    # Set plot titles
    titles = ['Input', 'Output-Generated', 'Original', 'Error']
    ranges_vmin_vmax = [[1.3, 4.2], [1.3, 4.2], [1.3, 4.2], [0.0, 0.021]]

    # Create a figure with a size of 10 inches by 4 inches
    fig = plt.figure(figsize=(15, 5))

    # plot images row by row
    for i in range(len(images)):
        # define subplot
        ax = fig.add_subplot(1, 4, 1 + i)
        # plot raw pixel data
        if i < 3:
            im = ax.imshow(images[i, :, :, 0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])
        else:
            im = ax.imshow(images[i, :, :, 0], cmap='viridis')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
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
        cbar.set_label('IC', fontsize=9)
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=6)
        cbar.update_ticks()

    plt.show()
