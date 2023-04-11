import numpy as np
import matplotlib.pyplot as plt
from functions.p2p_train_architecture import generate_real_samples, generate_fake_samples



# Save the generator model and check how good the generated image looks.
def summarize_performance(step, g_model, dataset, n_samples=3):
    print('... Saving a summary')
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    # plot real source images
    for i in range(n_samples):
        ax = plt.subplot(3, n_samples, 1 + i)
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
        ax = plt.subplot(3, n_samples, 1 + n_samples + i)
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
        ax = plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
        # plt.axis('off')
        plt.imshow(X_realB[i])
        if i == 0:
            plt.ylabel('Real Target', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xticklabels(['0', '64', '128', '192', '256'], fontsize=6)
        ax.set_yticks([0, 32, 64])
        ax.set_yticklabels(['0', '32', '64'], fontsize=6)

    filename1 = 'plot_%06d.png' % (step + 1)
    plt.savefig(filename1)
    plt.close()
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



# Plot the input, generated and original images
def plot_images(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Input', 'Output-Generated', 'Original']
    ranges_vmin_vmax = [[1.6, 4], [1.6, 4], [1.6, 4], [0, 2]]
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        plt.subplot(1, 3, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i,:,:,0], cmap='viridis')
        # show title
        plt.title(titles[i])
    plt.show()



# Plot the input, generated and original images
def plot_images_with_error(src_img, gen_img, tar_img):
    images = np.vstack((src_img, gen_img, tar_img, np.abs(gen_img-tar_img)))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Input', 'Output-Generated', 'Original', 'Error']
    ranges_vmin_vmax = [[1.6, 4], [1.6, 4], [1.6, 4], [0, 2]]
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        plt.subplot(1, 4, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i,:,:,0], cmap='viridis', vmin=ranges_vmin_vmax[i][0], vmax=ranges_vmin_vmax[i][1])
        # show title
        plt.title(titles[i])
    plt.show()