from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from tensorflow.keras.models import load_model

from functions.p2p_process_data import read_all_csv_files, apply_miss_rate_per_rf, preprocess_data
from functions.p2p_discriminator_architecture import define_discriminator
from functions.p2p_generator_architecture import define_generator
from functions.p2p_gan_architecture import define_gan
from functions.p2p_train_architecture import train
from functions.p2p_summary import plot_images


# Resizing images, if needed
SIZE_X = 256
SIZE_Y = 64
no_rows = SIZE_Y
no_cols = SIZE_X
path = 'C:\\inpt\\synthetic_data\\test'

#miss_rate = 0.9868
#min_distance = 51
miss_rate = 0.95
min_distance = 10


# Capture training image info as a list
tar_images = []

# Capture mask/label info as a list
src_images = []

# src_images = np.array(src_images)

all_csv = read_all_csv_files(path)
missing_data, full_data= apply_miss_rate_per_rf(all_csv, miss_rate, min_distance)
no_samples = len(all_csv)

missing_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in missing_data])
full_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in full_data])

tar_images = np.reshape(full_data, (no_samples, no_rows, no_cols, 1))
src_images = np.reshape(missing_data, (no_samples, no_rows, no_cols, 1))

n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(src_images[i], cmap='viridis')
# plot target image
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(tar_images[i], cmap='viridis')
plt.show()

#######################################################



# define input shape based on the loaded dataset
image_shape = src_images.shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

# Define data
# load and prepare training images
data = [src_images, tar_images]


# Preprocess data to change input range to values between -1 and 1
# This is because the generator uses tanh activation in the output layer
# And tanh ranges between -1 and 1
dataset = preprocess_data(data)



start1 = datetime.now()

train(d_model, g_model, gan_model, dataset, n_epochs=2, n_batch=1)
# Reports parameters for each batch (total 1600) for each epoch.
# For 10 epochs we should see 16000

stop1 = datetime.now()
# Execution time of the model
execution_time = stop1 - start1
print("Execution time is: ", execution_time)


#########################################
# Test trained model on a few images...
model = load_model('final_generator.h5')

[X1, X2] = dataset
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)
