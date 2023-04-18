import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions.p2p_summary import summarize_performance, plot_history
from functions.p2p_generate_samples import generate_real_samples, generate_fake_samples

results_dir_path = r'/scratch/fcamposmontero/results_p2p/512x32_e200_s2000'
#results_dir_path = r'C:\inpt\GAN_p2p\results\test'
#results_dir_path = r'/scratch/fcamposmontero/p2p_512x32_results_test'



# Train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]

    # unpack dataset
    trainA, trainB = dataset

    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    iterations = bat_per_epo * n_epochs

    # Create empty containers for the losses and accuracy
    d_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list()
    d_epoch_hist, g_epoch_hist, a1_epoch_hist, a2_epoch_hist = list(), list(), list(), list()

    # Manually enumerate epochs and batches
    for i in range(n_epochs):
        g_loss_all, d_loss_all, acc_real_all, acc_fake_all = 0.0, 0.0, 0.0, 0.0

        # Enumerate batches over the training set
        for j in range(bat_per_epo):
            # TRAIN THE DISCRIMINATOR
            # select a batch of real samples
            [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
            # update discriminator for real samples
            d_loss_real, d_acc_real = d_model.train_on_batch([X_realA, X_realB], y_real)
            # generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
            # update discriminator for generated samples
            d_loss_fake, d_acc_fake = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # TRAIN THE GENERATOR
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

            # Print losses on this batch
            print('Epoch>%d, Batch %d/%d, d=%.3f, g=%.3f, d=%.3f, d=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss, g_loss, d_acc_real, d_acc_fake))

            # Storing the losses and accuracy of the iterations.
            d_hist.append(d_loss)
            g_hist.append(g_loss)
            a1_hist.append(d_acc_real)
            a2_hist.append(d_acc_fake)

            g_loss_all += g_loss
            d_loss_all += d_loss
            acc_real_all += d_acc_real
            acc_fake_all += d_acc_fake

        epoch_loss_g = g_loss_all / j  # total generator loss for the epoch
        epoch_loss_d = d_loss_all / j  # total discriminator loss for the epoch
        epoch_acc_real = acc_real_all / j
        epoch_acc_fake = acc_fake_all / j
        g_epoch_hist.append(epoch_loss_g)
        d_epoch_hist.append(epoch_loss_d)
        a1_epoch_hist.append(epoch_acc_real)
        a2_epoch_hist.append(epoch_acc_fake)

        # Summarize model performance
        summarize_every_n_epochs = 1
        if i % summarize_every_n_epochs == 0:
            summarize_performance(i, g_model, dataset)
            plot_history(d_hist, g_hist, g_epoch_hist, d_epoch_hist,
                         a1_hist, a2_hist, a1_epoch_hist, a2_epoch_hist, i, n_epochs, iterations)


    # Save the generator model
    final_generator_path = os.path.join(results_dir_path, 'final_generator.h5')
    g_model.save(final_generator_path)
    plot_history(d_hist, g_hist, g_epoch_hist, d_epoch_hist,
                 a1_hist, a2_hist, a1_epoch_hist, a2_epoch_hist, i, n_epochs, iterations)
    # Save results to dataframe and CSV file
    df = pd.DataFrame({'disc_loss': d_hist, 'gen_loss': g_hist, 'acc_real': a1_hist, 'acc_fake': a2_hist})

    #csv_file = r'/scratch/fcamposmontero/p2p_512x32_results/results_loss.csv'
    csv_file = os.path.join(results_dir_path, 'results_loss.csv')
    df.to_csv(csv_file, index=False)


