import random
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.keras import Input
from tensorflow.optimizers import Adam
from tensorflow.models import Model
from tensorflow.models import Sequential
from tensorflow.datasets import mnist
from tensorflow.utils import to_categorical

from tensorflow.layers import Dense
from tensorflow.layers import Reshape
from tensorflow.layers import Flatten
from tensorflow.layers import Conv2D
from tensorflow.layers import Conv2DTranspose
from tensorflow.layers import LeakyReLU
from tensorflow.layers import Activation
from tensorflow.layers import Concatenate
from tensorflowkeras.layers import Dropout

from tensorflow.keras.datasets import mnist

# Check if GPUs are available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Constants and hyerparameters

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128


# import the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])



# Visualize the data
fig, ax = plt.subplots(1, 10, figsize=(8, 4))
for i in range(10):
    rand_idx = random.randint(0, 70000)
    ax[i].imshow(all_digits[rand_idx], cmap='Greys')
    ax[i].axis('off')
plt.show()

# Check the shape of the images and labels
print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")

# create the discriminator
def discriminator(in_shape=(28,28,1)):
    model = Sequential()
    model.add(InputLayer(28,28,1))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(pool_size=(3, 3)), strides=(2, 2))
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dropout(0.4))
    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=64))
    model.add(LeakyReLU(alpha=0.2))
    # add the FINAL dense layer for Binary Classification (fake/real)
    model.add(Dense(units=1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # return
    return model

def generator(latent_dim):
    model = Sequential()
