import numpy as np
import matplotlib.pyplot as plt
import random

from layers_functions.pert import pert
from layers_functions.layer_boundary import layer_boundary

x_max = 512             # length (x) of the model
z_max = 64              # depth (z) of the model
# Model coordinates
x_coord = np.arange(0, x_max, 1)       # array of x coordinates
z_coord = np.arange(0, z_max, 1)       # array of z coordinates
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")   # 2D mesh of coordinates x,z


matrix = np.zeros((z_max, x_max))
coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
values = np.zeros(coords_to_list.shape[0])

# Plot new matrix as image
fig, axs = plt.subplots(nrows=10, ncols=5, figsize=(20, 40))

for i in range(50):
    # Generate new y value for each plot
    y1 = layer_boundary(x_coord)
    y2 = layer_boundary(x_coord)
    y3 = layer_boundary(x_coord)
    y4 = layer_boundary(x_coord)
    boundaries = [y1, y2, y3, y4]
    boundaries = sorted(boundaries, key=lambda x: x[0])

    area_1 = []
    area_2 = []
    area_3 = []
    area_4 = []
    area_5 = []

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row <= boundaries[0][col]:
                area_1.append([col, row])
            elif row <= boundaries[1][col]:
                area_2.append([col, row])
            elif row <= boundaries[2][col]:
                area_3.append([col, row])
            elif row <= boundaries[3][col]:
                area_4.append([col, row])
            else:
                area_5.append([col, row])

    # Store lists in a list of lists
    lists = [area_1, area_2, area_3, area_4, area_5]

    # Sort the lists based on their first element in ascending order

    new_matrix = np.zeros_like(matrix)
    for j, lst in enumerate(lists):
        for coords in lst:
            new_matrix[coords[1], coords[0]] = j

    ax = axs[i // 5, i % 5]
    ax.imshow(new_matrix, cmap='viridis')

plt.tight_layout()
plt.show()