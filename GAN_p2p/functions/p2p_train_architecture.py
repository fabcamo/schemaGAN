import numpy as np
from functions.p2p_summary import summarize_performance, plot_history
from functions.p2p_generate_samples import generate_real_samples, generate_fake_samples


# Train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs

    d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list(), list()

    # Manually enumerate epochs and batches
    for i in range(n_epochs):
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

            # TRAIN THE GENERATOR
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

            # summarize performance
            # Print losses on this batch
            print('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))

            # Storing the losses and accuracy of the iterations.
            d1_hist.append(d_loss_real)
            d2_hist.append(d_loss_fake)
            d_hist = np.add(d1_hist, d2_hist).tolist()
            g_hist.append(g_loss)
            a1_hist.append(d_acc_real)
            a2_hist.append(d_acc_fake)

        # summarize model performance
        summarize_every_n_epochs = 5
        if i % summarize_every_n_epochs == 0:
            summarize_performance(i, g_model, dataset)
            plot_history(d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist)

    # Save the generator model
    final_generator_path = 'final_generator.h5'
    g_model.save(final_generator_path)
    plot_history(d1_hist, d2_hist, d_hist, g_hist, a1_hist, a2_hist)


