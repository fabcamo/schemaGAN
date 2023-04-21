import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from functions.p2p_generate_samples import generate_real_samples, generate_real_samples_fix, generate_fake_samples
from functions.p2p_process_data import reverse_normalization, reverse_IC_normalization

# For DelftBlue, un-comment this
#path_results = r'/scratch/fcamposmontero/results_p2p/512x32_e200_s2000'

# For local run, un-comment this
path_results = r'C:\inpt\GAN_p2p\results\test'


# Save the generator model and check how good the generated image looks.
def summarize_performance(step, g_model, dataset, n_samples=1):
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
                 maeF_epoch_hist, accR_hist, accF_hist, accR_epoch_hist, accF_epoch_hist, step, n_epochs, iterations):

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
    ax1.set_ylim([0, 4])
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

    # Calculate the Mean absolute error between the target image and the generated one
    mae = np.mean(np.absolute(tar_img - gen_img))

    # Stack all the images
    images = np.vstack((src_img, gen_img, tar_img, np.abs(gen_img-tar_img)))

    # Scale from [-1,1] to [0,255]
    images = reverse_IC_normalization(images)
    # Set plot titles
    titles = ['Input', 'Output-Generated', 'Original', f'Mean absolute error: {mae:.2f}']
    ranges_vmin_vmax = [[0, 4.5], [0, 4.5], [0, 4.5], [0, 4.5]]

    # Create a figure with a size of 10 inches by 4 inches
    fig = plt.figure(figsize=(10, 15))

    # plot images row by row
    for i in range(len(images)):
        # define subplot
        ax = fig.add_subplot(4, 1, 1 + i)
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



















def plot_images(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img))

    # Scale from [-1,1] to [0,255]
    images = reverse_IC_normalization(images)

    # Set plot titles
    titles = ['Input', 'Output-Generated', 'Original']
    ranges_vmin_vmax = [[1.3, 4.2], [1.3, 4.2], [1.3, 4.2]]

    # Create a figure with a size of 10 inches by 4 inches
    fig = plt.figure(figsize=(15, 10))

    # plot images row by row
    for i in range(len(images)):
        # define subplot
        ax = fig.add_subplot(1, 3, 1 + i)
        # plot raw pixel data
        #im = ax.imshow(images[i, :, :, 0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])
        im = ax.imshow(images[i, :, :, 0], cmap='viridis')
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

















# create a line plot of loss for the gan and save to file
def plot_history_old(d_hist, g_hist, g_epoch_hist, d_epoch_hist, maeR_hist, maeF_hist, maeR_epoch_hist, maeF_epoch_hist, step, n_epochs, iterations):
    # create figure for loss
    plt.figure(figsize=(10, 4))
    plt.plot(d_hist, label='disc', color='black')
    plt.plot(g_hist, label='gen', color='darkgray')
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # set x-axis limits
    plt.xlim([0, iterations])
    # Save plot to file
    plot_loss = os.path.join(path_results, 'plot_loss_{:06d}.png'.format(step + 1))
    plt.savefig(plot_loss)
    plt.close()

    # create figure for loss per epoch
    plt.figure(figsize=(10, 4))
    plt.plot(d_epoch_hist, label='disc', color='black')
    plt.plot(g_epoch_hist, label='gen', color='darkgray')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # set x-axis limits
    plt.xlim([0, n_epochs])
    # Save plot to file
    plot_loss = os.path.join(path_results, 'plot_loss_epoch_{:06d}.png'.format(step + 1))
    plt.savefig(plot_loss)
    plt.close()

    # create figure for accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(maeR_hist, label='acc-real', color='black', alpha=0.8)
    plt.plot(maeF_hist, label='acc-fake', color='darkgray', alpha=0.8)
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    # set x-axis limits
    plt.xlim([0, iterations])
    # set y-axis limits
    plt.ylim([0, 1])
    # Save plot to file
    plot_acc = os.path.join(path_results, 'plot_mae_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()

    # create figure for accuracy per epoch
    plt.figure(figsize=(10, 4))
    plt.plot(maeR_epoch_hist, label='acc-real', color='black', alpha=0.8)
    plt.plot(maeF_epoch_hist, label='acc-fake', color='darkgray', alpha=0.8)
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # set x-axis limits
    plt.xlim([0, n_epochs])
    # set y-axis limits
    plt.ylim([0, 1])
    # Save plot to file
    plot_acc = os.path.join(path_results, 'plot_mae_epoch_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()

    # create figure with two subplots for accuracy
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    # plot data on each subplot
    ax1.plot(maeR_hist, label='Real p/iter', color='black')
    ax1.plot(maeF_hist, label='Fake p/iter', color='dimgray')
    ax2.plot(maeR_epoch_hist, label='Real p/epoch', color='gray')
    ax2.plot(maeF_epoch_hist, label='Fake p/epoch', color='silver')
    # set labels and legends for each subplot
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('MAE')
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    # save plot to file
    plot_acc = os.path.join(path_results, 'plot_mae_combined_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()

    # create figure with two subplots for accuracy
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    # plot data on each subplot
    ax1.plot(d_hist, label='D p/iter', color='black')
    ax1.plot(g_hist, label='G p/iter', color='dimgray')
    ax2.plot(d_epoch_hist, label='D p/epoch', color='gray')
    ax2.plot(g_epoch_hist, label='G p/epoch', color='silver')
    # set labels and legends for each subplot
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Loss')
    ax1.set_ylim([0, 4])
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    # save plot to file
    plot_acc = os.path.join(path_results, 'plot_loss_combined_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()

