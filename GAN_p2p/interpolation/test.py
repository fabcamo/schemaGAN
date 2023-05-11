import matplotlib.pyplot as plt
import numpy as np

# Create a 2D array with some values
image = np.array([[0, 0, 0, 1, 0, 2, 0, 0, 8, 0],
                  [0, 0, 0, 1, 0, 2, 0, 0, 8, 0],
                  [0, 0, 0, 1, 0, 2, 0, 0, 8, 0],
                  [0, 0, 0, 1, 0, 2, 0, 0, 8, 0]])

# Create a masked array where the pixels with value 0 are masked
newimage = np.where(image == 0, np.nan, image)

# Plot the masked array with imshow
plt.imshow(newimage, cmap='viridis')
# Add a colorbar to the plot
plt.colorbar()

plt.show()

