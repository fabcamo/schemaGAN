import numpy as np


# Select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    #y = np.ones((n_samples, patch_shape, patch_shape*4, 1))
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_real_samples_fix(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose a fixed instance
    np.random.seed(11)
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape*4, 1))
    return [X1, X2], y


# Generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    #y = np.zeros((len(X), patch_shape, patch_shape*4, 1))
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

