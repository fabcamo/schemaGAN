import numpy as np
import matplotlib.pyplot as plt

# Generate random matrix
matrix = np.random.uniform(0, 1, size=(64, 512))

# Define sinusoidal path for split
x = np.linspace(0, matrix.shape[1], num=1000)


# Set the amplitude, period, phase shift, and vertical shift of the sine function
amplitude = 20
period = 500
phase_shift = 0
vertical_shift = 30


y = amplitude * np.sin(2 * np.pi * (x - phase_shift) / period) + vertical_shift

# Create mask to split matrix into two layers
mask = np.zeros_like(matrix, dtype=int)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if i <= y[int(j * 1000 / matrix.shape[1])]:
            mask[i,j] = 1
        else:
            mask[i,j] = 2

# Apply mask to matrix
masked_matrix = np.ma.masked_where(mask == 0, mask)
plt.imshow(masked_matrix, cmap='rainbow', interpolation='nearest')
plt.show()
