import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from schemaGAN.functions.utils import reverse_IC_normalization
from schemaGAN.functions.utils import generate_real_samples, generate_real_samples_fix, generate_fake_samples

# Set the font family to Arial
plt.rcParams['font.family'] = 'Arial'



def summarize_performance(step, g_model, dataset, path_results):
    """
    Summarize and save the performance of the GAN during training.

    Args:
        step (int): The current step/epoch of training.
        g_model (tensorflow.keras.Model): The generator model.
        dataset (tuple of numpy.ndarray): A tuple containing two arrays (trainA and trainB).
        path_results (str): Path to save the results and model.

    """

    print('... Saving a summary')

    # Generate a real and fake image for comparison
    [src_image, tar_image], _ = generate_real_samples_fix(dataset, 1, 1, random_seed=14)
    gen_image, _ = generate_fake_samples(g_model, src_image, 1)

    # Plot and save images
    plot_images_error(src_image, gen_image, tar_image)
    plot_results_name = os.path.join(path_results, 'res_{:06d}.png'.format(step + 1))
    plt.savefig(plot_results_name)
    plt.close()

    # Save the generator model
    model_name = os.path.join(path_results, 'model_{:06d}.h5'.format(step + 1))
    g_model.save(model_name)
    print('> Saved: {} and {}'.format(plot_results_name, model_name))



def plot_histories(d_hist, g_hist, g_epoch_hist, d_epoch_hist, maeR_hist, maeF_hist, maeR_epoch_hist,
                   maeF_epoch_hist, accR_hist, accF_hist, accR_epoch_hist, accF_epoch_hist,
                   step, n_epochs, iterations, path_results):
    """
    Plot and save the history of losses and metrics during training.

    Args:
        d_hist (list): List of discriminator losses per iteration.
        g_hist (list): List of generator losses per iteration.
        g_epoch_hist (list): List of generator losses per epoch.
        d_epoch_hist (list): List of discriminator losses per epoch.
        maeR_hist (list): List of mean absolute error (MAE) for real images per iteration.
        maeF_hist (list): List of mean absolute error (MAE) for fake images per iteration.
        maeR_epoch_hist (list): List of MAE for real images per epoch.
        maeF_epoch_hist (list): List of MAE for fake images per epoch.
        accR_hist (list): List of accuracies for real images per iteration.
        accF_hist (list): List of accuracies for fake images per iteration.
        accR_epoch_hist (list): List of accuracies for real images per epoch.
        accF_epoch_hist (list): List of accuracies for fake images per epoch.
        step (int): The current step/epoch of training.
        n_epochs (int): Total number of epochs.
        iterations (int): Total number of iterations.
        path_results (str): Path to save the plot images.

    Returns:
        None
    """

    # Plot loss history
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    ax1.plot(d_hist, label='D loss', color='black')
    ax1.plot(g_hist, label='G loss', color='dimgray')
    ax2.plot(d_epoch_hist, label='D avg', color='gray')
    ax2.plot(g_epoch_hist, label='G avg', color='silver')
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Loss')
    ax1.set_ylim([0, 6])
    fig.legend(loc='upper right', bbox_to_anchor=(1.03, 0.85))
    plot_loss = os.path.join(path_results, 'plot_loss{:06d}.png'.format(step + 1))
    plt.savefig(plot_loss, bbox_inches='tight')
    plt.close()

    # Plot MAE history
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    ax1.plot(maeR_hist, label='Real', color='black')
    ax1.plot(maeF_hist, label='Fake', color='dimgray')
    ax2.plot(maeR_epoch_hist, label='Real avg', color='gray')
    ax2.plot(maeF_epoch_hist, label='Fake avg', color='silver')
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Mean absolute error')
    ax1.set_ylim([0, 1])
    fig.legend(loc='upper right', bbox_to_anchor=(1.03, 0.85))
    plot_mae = os.path.join(path_results, 'plot_mae{:06d}.png'.format(step + 1))
    plt.savefig(plot_mae, bbox_inches='tight')
    plt.close()

    # Plot accuracy history
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twiny()
    ax1.plot(accR_hist, label='Real', color='black')
    ax1.plot(accF_hist, label='Fake', color='dimgray')
    ax2.plot(accR_epoch_hist, label='Real avg', color='gray')
    ax2.plot(accF_epoch_hist, label='Fake avg', color='silver')
    ax1.set_xlabel('Iterations')
    ax1.set_xlim([0, iterations])
    ax2.set_xlabel('Epochs')
    ax2.set_xlim([0, n_epochs])
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    fig.legend(loc='upper right', bbox_to_anchor=(1.03, 0.85))
    plot_acc = os.path.join(path_results, 'plot_acc{:06d}.png'.format(step + 1))
    plt.savefig(plot_acc, bbox_inches='tight')
    plt.close()



def plot_images_error(src_img, gen_img, tar_img):
    """
    Plot the source image, generated image, target image, and their absolute error.

    Args:
        src_img (numpy.ndarray): Source image.
        gen_img (numpy.ndarray): Generated image.
        tar_img (numpy.ndarray): Target image.

    Returns:
        None
    """

    # Set the font family to Arial
    plt.rcParams['font.family'] = 'Arial'

    # Scale from [-1,1] to [0,255]
    src_img = reverse_IC_normalization(src_img)
    tar_img = reverse_IC_normalization(tar_img)
    gen_img = reverse_IC_normalization(gen_img)

    # Calculate the Mean absolute error between the target image and the generated one
    mae = np.mean(np.abs(tar_img - gen_img))

    # Stack all the images
    images = np.vstack((src_img, gen_img, tar_img, np.abs(gen_img - tar_img)))

    # Set plot titles
    titles = ['CPT-like input', 'SchemaGAN Generation', 'Original Schematization', f'Mean absolute error: {mae:.2f}']
    # Set the cbar range
    ranges_vmin_vmax = [[0, 4.5], [1, 4.5], [1, 4.5], [0, 1]]
    # Set the cbar titles
    cbar_titles = ['IC', 'IC', 'IC', 'IC error']

    # Create a figure with a size of 10 inches by 15 inches
    fig = plt.figure(figsize=(10, 15))
    # Plot images row by row
    for i in range(len(images)):
        # Define subplot
        ax = fig.add_subplot(4, 1, 1 + i)
        im = ax.imshow(images[i, :, :, 0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])

        # Set title with fontsize
        ax.set_title(titles[i], fontsize=10)

        # Set tick_params with fontsize
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='both', which='minor', labelsize=9)

        # Set x and y labels
        ax.set_xlabel('Distance', fontsize=9)
        ax.set_ylabel('Depth', fontsize=9)

        # Manually set tick mark spacing
        ax.set_xticks(np.arange(0, images.shape[2], 40))
        ax.set_yticks(np.arange(0, images.shape[1], 20))

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.16, shrink=0.7)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(cbar_titles[i], fontsize=9)
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=6)
        cbar.update_ticks()
