from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from numpy.random import randn

# For a single image
# example of generating an image for a specific point in the latent space

"""
# load model
model = load_model('C:\\inpt\\GAN_vanilla\\results\\tets\\mnist_final_generator.h5')

#To create same image, suppy same vector each time
# all 0s
#vector = asarray([[0. for _ in range(100)]])  #Vector of all zeros

#To create random images each time...
vector = randn(128) #Vector of random numbers (creates a column, need to reshape)
vector = vector.reshape(1, 128)

# generate image
X = model.predict(vector)

# plot the result
pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.show()

"""



#"""

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		plt.subplot(n, n, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i, :, :, 0], cmap='viridis')
	plt.show()

# load model
model = load_model('C:\\inpt\\GAN_vanilla\\results\\results_1266e\\model_001271.h5')
# generate images
#Generate 16 images, each image provide a vector of size 100 as input
latent_points = generate_latent_points(128, 16)
# generate images
X = model.predict(latent_points)
# plot the result
save_plot(X, 4)  #Plot 4x4 grid (Change to 5 if generating 25 images)

