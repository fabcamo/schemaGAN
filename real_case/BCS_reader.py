import numpy as np
import matplotlib.pyplot as plt

# Load the data from the file
file_path = r"N:\Deltabox\Postbox\Campos Montero, Fabian\83_correct_wpadding\bcs_best_estimate.txt"  # Change this to the path of your file
data = np.loadtxt(file_path)

# Reshape the data into the desired shape (512x32)
data_reshaped = data[:512, :32].T
# Invert the data along the x-axis
data_reshaped = data_reshaped[::-1, :]

# Plot and save the image
plt.imshow(data_reshaped, cmap='viridis', aspect='equal')  # Set aspect to 'equal' to keep proportions
plt.axis('on')  # Turn on axis labels
plt.colorbar()  # Show color bar for value mapping

# Save the image
plt.savefig('output_image_with_axis.png', bbox_inches='tight', pad_inches=0)
plt.show()


