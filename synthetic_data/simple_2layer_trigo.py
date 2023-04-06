import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial import Delaunay

from layers_functions.rf_model import random_field_generator, random_field_generator2
from layers_functions.rf_2D_polygons import generate_2D_polygons
from layers_functions.pert import pert

x_max = 512             # length (x) of the model
z_max = 64              # depth (z) of the model
# Model coordinates
x_coord = np.arange(0, x_max, 1)       # array of x coordinates
z_coord = np.arange(0, z_max, 1)       # array of z coordinates
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
# generate the random field models for different materials
srf_sand = random_field_generator(0.3, 1.5, aniso_x, aniso_z, ndim, seed+1)
srf_clay = random_field_generator(0.3, 2.1, aniso_x, aniso_z, ndim, seed+2)
srf_silt = random_field_generator(0.5, 3.2, aniso_x, aniso_z, ndim, seed+3)
srf_other = random_field_generator2(ndim, var, len_scale, anis, angles, mean, seed+4)
# store the random field models inside layers
layers = [srf_sand, srf_silt, srf_clay, srf_sand, srf_clay, srf_silt, srf_sand]

##############################################################################################


matrix = np.zeros((z_max, x_max))
print('matrix shape 0>', matrix.shape[0])
print('matrix shape 1>', matrix.shape[1])
coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
print('coords to list')
print(coords_to_list)
values = np.zeros(coords_to_list.shape[0])

amplitude = pert(2,10,90)
period = pert(200, 1000, 6000)
phase_shift = np.random.uniform(low=0, high=500)
vertical_shift = 30
func = random.choice([np.sin, np.cos])
y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

above_list = []
below_list = []

# Create mask to split matrix into two layers
for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        if row <= y[col]:
            above_list.append([col,row])
        else:
            below_list.append([col,row])



# Store lists in a list of lists
above_array = np.array(above_list)
below_array = np.array(below_list)

lists = [above_array, below_array]


new_matrix = np.zeros_like(matrix)
for i, lst in enumerate(lists):
    mask = (coords_to_list[:, None] == lists[i]).all(2).any(1)
    layer_coordinates = coords_to_list[mask]
    layer_IC = layers[i](layer_coordinates.T)
    values[mask] = layer_IC

# store the results in a dataframe
df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})
grouped = df.groupby('x')
df_pivot = df.pivot(index="z", columns="x", values="IC")
plt.imshow(df_pivot, cmap='viridis')
plt.show()