import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions.p2p_summary import summarize_performance, plot_history
from functions.p2p_generate_samples import generate_real_samples, generate_fake_samples


# Train pix2pix models
def train(path_results, d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]

    # Unpack dataset
    trainA, trainB = dataset

    # Calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # Calculate the number of training iterations
    iterations = bat_per_epo * n_epochs

    # Create empty containers for the losses and accuracy
    d_hist, g_hist, maeR_hist, maeF_hist, accR_hist, accF_hist = list(), list(), list(), list(), list(), list()
    d_epoch_hist, g_epoch_hist, maeR_epoch_hist, maeF_epoch_hist, accR_epoch_hist, accF_epoch_hist = list(), list(), list(), list(), list(), list()

    # Manually enumerate epochs and batches
    for i in range(n_epochs):
        g_loss_all, d_loss_all, mae_real_all, mae_fake_all, acc_real_all, acc_fake_all = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Enumerate batches over the training set
        for j in range(bat_per_epo):
            # TRAIN THE DISCRIMINATOR
            # select a batch of real samples
            [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
            # update discriminator for real samples
            d_loss_real, d_mae_real, d_acc_real = d_model.train_on_batch([X_realA, X_realB], y_real)
            # generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
            # update discriminator for generated samples
            d_loss_fake, d_mae_fake, d_acc_fake = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # TRAIN THE GENERATOR
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

            # Print losses on this batch
            print('Epoch>%d, Batch %d/%d, Dloss=%.3f, Gloss=%.3f, DmaeReal=%.3f, DmaeFake=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss, g_loss, d_mae_real, d_mae_fake))

            # Storing the losses and accuracy of the iterations.
            d_hist.append(d_loss)
            g_hist.append(g_loss)
            maeR_hist.append(d_mae_real)
            maeF_hist.append(d_mae_fake)
            accR_hist.append(d_acc_real)
            accF_hist.append(d_acc_fake)

            g_loss_all += g_loss
            d_loss_all += d_loss
            mae_real_all += d_mae_real
            mae_fake_all += d_mae_fake
            acc_real_all += d_acc_real
            acc_fake_all += d_acc_fake

        epoch_loss_g = g_loss_all / j  # total generator loss for the epoch
        epoch_loss_d = d_loss_all / j  # total discriminator loss for the epoch
        epoch_mae_real = mae_real_all / j
        epoch_mae_fake = mae_fake_all / j
        epoch_acc_real = acc_real_all / j
        epoch_acc_fake = acc_fake_all / j
        g_epoch_hist.append(epoch_loss_g)
        d_epoch_hist.append(epoch_loss_d)
        maeR_epoch_hist.append(epoch_mae_real)
        maeF_epoch_hist.append(epoch_mae_fake)
        accR_epoch_hist.append(epoch_acc_real)
        accF_epoch_hist.append(epoch_acc_fake)

        # Summarize model performance
        summarize_every_n_epochs = 1
        if i % summarize_every_n_epochs == 0:
            summarize_performance(i, g_model, dataset, path_results)
            plot_history(d_hist, g_hist, g_epoch_hist, d_epoch_hist,
                         maeR_hist, maeF_hist, maeR_epoch_hist, maeF_epoch_hist,
                         accR_hist, accF_hist, accR_epoch_hist, accF_epoch_hist,
                         i, n_epochs, iterations, path_results)


    # Save the generator model
    final_generator_path = os.path.join(path_results, 'final_generator.h5')
    g_model.save(final_generator_path)
    plot_history(d_hist, g_hist, g_epoch_hist, d_epoch_hist,
                 maeR_hist, maeF_hist, maeR_epoch_hist, maeF_epoch_hist,
                 accR_hist, accF_hist, accR_epoch_hist, accF_epoch_hist,
                 i, n_epochs, iterations, path_results)

    # Save results to dataframe and CSV file
    df = pd.DataFrame({'disc_loss': d_hist, 'gen_loss': g_hist, 'mae_real': maeR_hist,
                       'mae_fake': maeF_hist, 'acc_real': accR_hist, 'acc_fake': accF_hist})

    csv_file = os.path.join(path_results, 'results_loss.csv')
    df.to_csv(csv_file, index=False)


