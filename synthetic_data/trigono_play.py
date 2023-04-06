import numpy as np
import matplotlib.pyplot as plt
from layers_functions.pert import pert
# Generate random matrix
matrix = np.zeros((64, 512))

# Define sinusoidal paths for split
x = np.linspace(0, matrix.shape[1], num=1000)

amplitude = 10
period = 200
phase_shift = 0
vertical_shift = [50, 55, 20]  # different vertical shifts for the three lines

y1 = amplitude * np.sin(2 * np.pi * (x - phase_shift) / period) + vertical_shift[0]
y2 = amplitude * np.sin(2 * np.pi * (x - phase_shift) / period) + vertical_shift[1]
y3 = amplitude * np.sin(2 * np.pi * (x - phase_shift) / period) + vertical_shift[2]

# Create lists of coordinates for each section
above_list = []
between_list = []
below_list = []

for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        if row <= y1[col]:
            above_list.append([col, row])
        elif row <= y2[col]:
            between_list.append([col, row])
        else:
            below_list.append([col, row])

# Store lists in a list of lists
lists = [above_list, between_list, below_list]

# Plot new matrix as image
new_matrix = np.zeros_like(matrix)
for i, lst in enumerate(lists):
    for coords in lst:
        new_matrix[coords[1], coords[0]] = i

plt.imshow(new_matrix, cmap='viridis')
plt.show()
