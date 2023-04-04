import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial import Delaunay

from layers_functions.rf_model import random_field_generator, random_field_generator2
from layers_functions.rf_2D_polygons import generate_2D_polygons



##### MAIN VARIABLES ######################################################################################
n_layers = 5
x_max = 512             # length (x) of the model
z_max = 64              # depth (z) of the model
# Model coordinates
x_coord = np.linspace(0, x_max, x_max, dtype=int)       # array of x coordinates
z_coord = np.linspace(0, z_max, z_max, dtype=int)       # array of z coordinates
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")   # 2D mesh of coordinates x,z

##### RANDOM FIELD PARAMETERS ##############################################################################
std_value = 0.4         # standard deviation value
mean = 2                # mean value
aniso_x = 40            # anisotropy in X
aniso_z = 20            # anisotropy in Z
anis = 1 / 2            # anisotropy
ndim = 2                # number of dimensions
var = 0.8               # variance
len_scale = 32          # main length scale
angles = 0              # angle of rotation
seed = 20170519         # seed
############################################################################################################

matrix = np.zeros((z_max, x_max))
coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
values = np.zeros(coords_to_list.shape[0])

amplitude = 10
period = 200
phase_shift = 0
vertical_shift = 30
y = amplitude * np.sin(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

above_list = []
below_list = []


# Create mask to split matrix into two layers
for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        if row <= y[col]:
            above_list.append([row,col])
        else:
            below_list.append([row,col])

# Store lists in a list of lists
lists = [above_list, below_list]
above_array = np.array(above_list)
below_array = np.array(below_list)

# generate the random field models for different materials
srf_sand = random_field_generator(0.3, 1.5, aniso_x, aniso_z, ndim, seed+1)
srf_clay = random_field_generator(0.3, 2.1, aniso_x, aniso_z, ndim, seed+2)
layers = [srf_sand, srf_clay]
#random.shuffle(layers)  # shuffle the order of the list of materials for each loop


# evaluate the corresponding SRF to the current polygon
layer_IC_1 = layers[0](above_array.T)
layer_IC_2 = layers[1](below_array.T)

mask = np.isin(coords_to_list, above_array)



# Plot new matrix as image
new_matrix = np.zeros_like(matrix)
for i, lst in enumerate(lists):
    for coords in lst:
        new_matrix[coords[0], coords[1]] = i

plt.imshow(new_matrix, cmap='viridis')
plt.show()




