import matplotlib.pyplot as plt
import numpy as np

# Generate sample images
n_images = 25
image_shape = (28, 28)
X = np.random.rand(n_images, *image_shape)

# Create figure and axes objects
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8, 8))

# Plot images on axes
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i], cmap='gray')
    ax.axis('off')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)

# Display the plot
plt.show()