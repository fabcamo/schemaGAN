import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from GAN_p2p.functions.p2p_generate_samples import generate_real_samples, generate_real_samples_fix, generate_fake_samples
from GAN_p2p.functions.p2p_process_data import reverse_normalization, reverse_IC_normalization


# Save the generator model and check how good the generated image looks.
def summarize_performance(step, g_model, dataset, path_results):
    print('... Saving a summary')

    [src_image, tar_image], _ = generate_real_samples_fix(dataset, 1, 1)
    gen_image, _ = generate_fake_samples(g_model, src_image, 1)
    plot_images_error(src_image, gen_image, tar_image)
    plot_results_name = os.path.join(path_results, 'res_{:06d}.png'.format(step + 1))
    plt.savefig(plot_results_name)
    plt.close()

    # save the generator model
    os.path.join(path_results, 'model_%06d.h5' % (step + 1))
    model_name = os.path.join(path_results, 'model_%06d.h5' % (step + 1))
    g_model.save(model_name)
    print('>Saved: %s and %s' % (plot_results_name, model_name))



# create a line plot of loss for the gan and save to file
def plot_history(d_hist, g_hist, g_epoch_hist, d_epoch_hist, maeR_hist, maeF_hist, maeR_epoch_hist,
                 maeF_epoch_hist, accR_hist, accF_hist, accR_epoch_hist, accF_epoch_hist,
                 step, n_epochs, iterations, path_results):

    # create figure with two subplots for loss in generator and discriminator
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    # plot data on each subplot
    ax1.plot(d_hist, label='D loss', color='black')
    ax1.plot(g_hist, label='G loss', color='dimgray')
    ax2.plot(d_epoch_hist, label='D avg', color='gray')
    ax2.plot(g_epoch_hist, label='G avg', color='silver')
    # set labels and legends for each subplot
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Loss')
    ax1.set_ylim([0, 6])
    # add grid lines
    #ax1.grid(True)
    #ax2.grid(True)
    # combine both legends in a single box
    fig.legend(loc='upper right', bbox_to_anchor=(1.03, 0.85))
    # save plot to file
    plot_acc = os.path.join(path_results, 'plot_loss{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc, bbox_inches='tight')
    plt.close()

    # create figure with two subplots for mean absolute error in discriminator real vs fake
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    # plot data on each subplot
    ax1.plot(maeR_hist, label='Real', color='black')
    ax1.plot(maeF_hist, label='Fake', color='dimgray')
    ax2.plot(maeR_epoch_hist, label='Real avg', color='gray')
    ax2.plot(maeF_epoch_hist, label='Fake avg', color='silver')
    # set labels and legends for each subplot
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Mean absolute error')
    ax1.set_ylim([0, 1])
    # add grid lines
    #ax1.grid(True)
    #ax2.grid(True)
    # combine both legends in a single box
    fig.legend(loc='upper right', bbox_to_anchor=(1.03, 0.85))
    # save plot to file
    plot_acc = os.path.join(path_results, 'plot_mae{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc, bbox_inches='tight')
    plt.close()

    # create figure with two subplots for mean absolute error in discriminator real vs fake
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    # plot data on each subplot
    ax1.plot(accR_hist, label='Real', color='black')
    ax1.plot(accF_hist, label='Fake', color='dimgray')
    ax2.plot(accR_epoch_hist, label='Real avg', color='gray')
    ax2.plot(accF_epoch_hist, label='Fake avg', color='silver')
    # set labels and legends for each subplot
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    # add grid lines
    #ax1.grid(True, linewidth=0.1)
    #ax2.grid(True, linewidth=0.1)
    # combine both legends in a single box
    fig.legend(loc='upper right', bbox_to_anchor=(1.03, 0.85))
    # save plot to file
    plot_acc = os.path.join(path_results, 'plot_acc{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc, bbox_inches='tight')
    plt.close()


def plot_images_error(src_img, gen_img, tar_img):

    # Scale from [-1,1] to [0,255]
    src_img = reverse_IC_normalization(src_img)
    tar_img = reverse_IC_normalization(tar_img)
    gen_img = reverse_IC_normalization(gen_img)

    # Calculate the Mean absolute error between the target image and the generated one
    mae = np.mean(np.abs(tar_img - gen_img))

    # Stack all the images
    images = np.vstack((src_img, gen_img, tar_img, np.abs(gen_img-tar_img)))

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
        im = ax.imshow(images[i, :, :, 0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])
        #im = ax.imshow(images[i, :, :, 0], cmap='viridis')
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


