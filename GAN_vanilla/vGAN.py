import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.layers import InputLayer, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Import data from MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
all_images = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Check the image size
print('The shape of a single image is: ',all_images[0].shape)

# Plot to check the data
fig, axes = plt.subplots(1, 6, figsize=(10, 3))
for i in range(6):
    axes[i].imshow(all_images[i], cmap='Greys')
    axes[i].axis('off')
plt.show()      # To show the image
plt.clf()       # To clear the memory

