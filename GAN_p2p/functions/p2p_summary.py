import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functions.p2p_generate_samples import generate_real_samples, generate_real_samples_fix, generate_fake_samples
from functions.p2p_process_data import reverse_normalization

results_dir_path = r'/scratch/fcamposmontero/p2p_512x32_results10'
#results_dir_path = r'C:\inpt\GAN_p2p\results\test'
#results_dir_path = r'/scratch/fcamposmontero/p2p_512x32_results_test'


# Save the generator model and check how good the generated image looks.
def summarize_performance(step, g_model, dataset, n_samples=1):
    print('... Saving a summary')

    [src_image, tar_image], _ = generate_real_samples_fix(dataset, 1, 1)
    gen_image, _ = generate_fake_samples(g_model, src_image, 1)
    plot_images_error(src_image, gen_image, tar_image)
    plot_filename1 = os.path.join(results_dir_path, 'res_{:06d}.png'.format(step + 1))
    plt.savefig(plot_filename1)
    plt.close()

    # save the generator model
    os.path.join(results_dir_path, 'model_%06d.h5' % (step + 1))
    model_name = os.path.join(results_dir_path, 'model_%06d.h5' % (step + 1))
    g_model.save(model_name)
    print('>Saved: %s and %s' % (plot_filename1, model_name))



# create a line plot of loss for the gan and save to file
def plot_history_old(d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist):
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



# create a line plot of loss for the gan and save to file
def plot_history(d_hist, g_hist, g_epoch_hist, d_epoch_hist, a1_hist, a2_hist, a1_epoch_hist, a2_epoch_hist, step, n_epochs, iterations):
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
    plot_loss = os.path.join(results_dir_path, 'plot_loss_{:06d}.png'.format(step + 1))
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
    plot_loss = os.path.join(results_dir_path, 'plot_loss_epoch_{:06d}.png'.format(step + 1))
    plt.savefig(plot_loss)
    plt.close()

    # create figure for accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(a1_hist, label='acc-real', color='black', alpha=0.8)
    plt.plot(a2_hist, label='acc-fake', color='darkgray', alpha=0.8)
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    # set x-axis limits
    plt.xlim([0, iterations])
    # set y-axis limits
    plt.ylim([0, 1])
    # Save plot to file
    plot_acc = os.path.join(results_dir_path, 'plot_acc_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()

    # create figure for accuracy per epoch
    plt.figure(figsize=(10, 4))
    plt.plot(a1_epoch_hist, label='acc-real', color='black', alpha=0.8)
    plt.plot(a2_epoch_hist, label='acc-fake', color='darkgray', alpha=0.8)
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # set x-axis limits
    plt.xlim([0, n_epochs])
    # set y-axis limits
    plt.ylim([0, 1])
    # Save plot to file
    plot_acc = os.path.join(results_dir_path, 'plot_acc_epoch_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()

    # create figure with two subplots for accuracy
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    # plot data on each subplot
    ax1.plot(a1_hist, label='Real p/iter', color='black')
    ax1.plot(a2_hist, label='Fake p/iter', color='dimgray')
    ax2.plot(a1_epoch_hist, label='Real p/epoch', color='gray')
    ax2.plot(a2_epoch_hist, label='Fake p/epoch', color='silver')
    # set labels and legends for each subplot
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    # save plot to file
    plot_acc = os.path.join(results_dir_path, 'plot_acc_combined_{:06d}.png'.format(step + 1))
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
    plot_acc = os.path.join(results_dir_path, 'plot_loss_combined_{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc)
    plt.close()





def plot_images(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img))

    # Scale from [-1,1] to [0,255]
    images = reverse_normalization(images)

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

    # Scale from [-1,1] to [0,255]
    images = reverse_normalization(images)
    # Set plot titles
    titles = ['Input', 'Output-Generated', 'Original', 'MAE']
    ranges_vmin_vmax = [[1.3, 4.2], [1.3, 4.2], [1.3, 4.2], [110, 130]]

    # Create a figure with a size of 10 inches by 4 inches
    fig = plt.figure(figsize=(10, 15))

    # plot images row by row
    for i in range(len(images)):
        # define subplot
        ax = fig.add_subplot(4, 1, 1 + i)
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













# Save the generator model and check how good the generated image looks.
def summarize_performance_old(step, g_model, dataset, n_samples=1):
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


    filename1 = os.path.join(results_dir_path, 'plot_{:06d}.png'.format(step + 1))
    plt.savefig(filename1)
    plt.close()

    [src_image, tar_image], _ = generate_real_samples(dataset, 1, 1)
    gen_image, _ = generate_fake_samples(g_model, src_image, 1)
    plot_images_error(src_image, gen_image, tar_image)
    plot_filename1 = os.path.join(results_dir_path, 'res_{:06d}.png'.format(step + 1))
    plt.savefig(plot_filename1)
    plt.close()

    # save the generator model
    filename2 = 'model_%06d.h5' % (step + 1)
    #g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))