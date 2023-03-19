import os
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

from vGAN_models import discriminator_model, generator_model, GAN_model
from vGAN_models import normalize_data, train


# Define the PATH
path = r'D:\inpt\GAN_vanilla\results'
# Define the path and the results name file
results_file_path = os.path.join(path, 'results_summary.txt')

# Check the time and start the timers
time_current = time.strftime("%d/%m/%Y %H:%M:%S")
time_start = time.time() # start the timer



#################################################################################################################
#   DEFINE THE HYER-PARAMETERS
#################################################################################################################
no_channels = 1         # number of channels in the image
latent_dim = 128        # user defined number as input to the generator
n_samples = 25          # number of samples

n_epochs = 2000         # number of epochs
n_batch = 32            # number of samples in batch

# Get the shape of the input data
all_images, all_labels = load_data()
input_shape = all_images[0].shape



#################################################################################################################
#   TRAIN THE GAN
#################################################################################################################
# Create the Discriminator
discriminator = discriminator_model(input_shape)

# Create the Generator
generator = generator_model(latent_dim)

# Create the GAN
gan_model = GAN_model(generator, discriminator)

# Load the dataset of images
dataset = normalize_data()

# Train the model
train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs, n_batch)

time_end = time.time() # End the timer



#################################################################################################################
#   GENERATE THE SUMMARY OF THE MODEL
#################################################################################################################
time_total = time_start - time_end # Calculate the run time

# Format time taken to run into> Hours : Minutes : Seconds
hours = int(time_total // 3600)
minutes = int((time_total % 3600) // 60)
seconds = int(time_total % 60)
time_str = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

# Save the time to the summary file
with open(results_file_path, "a") as file:
    file.write("Executed on: {}\n".format(time_current))
    file.write("Time taken: {}\n".format(time_str))
    file.write("The shape of a single input image is: {}\n".format(input_shape))

# Check if GPUs are available and write it to the summary
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    num_gpus = len(gpus)
    with open(results_file_path, "a") as file:
        file.write("Running TensorFlow with {} GPU(s)\n".format(num_gpus))
else:
    with open(results_file_path, "a") as file:
        file.write("Running TensorFlow on CPU\n")

with open(results_file_path, "a") as file:
    file.write("The shape of a single input image is: {}\n".format(input_shape))

print("Time taken: {}".format(time_str))
print("Results saved to ", results_file_path)



#################################################################################################################
#   LOAD THE MODEL AND GENERATE SOME IMAGES TO CHECK
#################################################################################################################
# Plot generated images
def show_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
    plt.show()

# Load model
model = load_model('D:\\inpt\\GAN_vanilla\\results\\mnist_final_generator.h5')  # Model trained for 100 epochs
# Generate images
latent_points = generate_noise_vectors(latent_dim, 25)  # Latent dim and n_samples
# Generate images
X = model.predict(latent_points)
# Scale from [-1,1] to [0,1]
X = (X + 1) / 2.0

X = (X * 255).astype(np.uint8)

# Plot the result
show_plot(X, 5)

