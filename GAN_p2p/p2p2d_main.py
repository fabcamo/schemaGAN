import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint
from numpy import vstack

from p2p2d_model_256 import read_all_csv_files, apply_miss_rate_per_rf
from p2p2d_model_256 import define_discriminator, define_generator, define_gan, train

# Resizing images, if needed
SIZE_X = 256
SIZE_Y = 64
no_rows = SIZE_Y
no_cols = SIZE_X
path = 'D:\\inpt\\synthetic_data\\test'

miss_rate = 0.90
min_distance = 6

# n_classes=4 #Number of classes for segmentation

# Capture training image info as a list
tar_images = []

# Capture mask/label info as a list
src_images = []

# src_images = np.array(src_images)

all_csv = read_all_csv_files(path)
missing_data, full_data= apply_miss_rate_per_rf(all_csv, miss_rate, min_distance)


missing_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in missing_data])
full_data = np.array([np.reshape(i, (no_rows, no_cols)).astype(np.float32) for i in full_data])

tar_images = np.reshape(full_data, (5, no_rows, no_cols, 1))
src_images = np.reshape(missing_data, (5, no_rows, no_cols, 1))

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
def preprocess_data(data):
    # load compressed arrays
    # unpack arrays
    X1, X2 = data[0], data[1]
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


dataset = preprocess_data(data)



start1 = datetime.now()

train(d_model, g_model, gan_model, dataset, n_epochs=50, n_batch=1)
# Reports parameters for each batch (total 1600) for each epoch.
# For 10 epochs we should see 16000

stop1 = datetime.now()
# Execution time of the model
execution_time = stop1 - start1
print("Execution time is: ", execution_time)

# Reports parameters for each batch (total 1096) for each epoch.
# For 10 epochs we should see 10960
g_model.save('256by64_.h5')
#########################################
# Test trained model on a few images...



#model = load_model('sandstone_50epochs.h5')


# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
    images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Input-segm-img', 'Output-Generated', 'Original']
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        plt.subplot(1, 3, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[i, :, :, 0], cmap='gray')
        # show title
        plt.title(titles[i])
    plt.show()


[X1, X2] = dataset
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)

###########################################
test_src_img = cv2.imread("sandstone/test_mask.tif", 1)
test_src_img = cv2.resize(test_src_img, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
test_src_img = (test_src_img - 127.5) / 127.5
test_src_img = np.expand_dims(test_src_img, axis=0)

# generate image from source
gen_test_image = model.predict(test_src_img)

# pyplot.imshow(test_src_img[0, :,:,0], cmap='gray')
plt.imshow(gen_test_image[0, :, :, 0], cmap='gray')