import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from layers_functions.layer_boundary import layer_boundary
from layers_functions.generate_rf import random_field_generator, generate_rf_group


x_max = 512             # length (x) of the model
z_max = 64              # depth (z) of the model
# Model coordinates
x_coord = np.arange(0, x_max, 1)       # array of x coordinates
z_coord = np.arange(0, z_max, 1)       # array of z coordinates
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")   # 2D mesh of coordinates x,z


##### RANDOM FIELD PARAMETERS ##############################################################################
std_value = 0.3
mean = 2.1
aniso_x = 40            # anisotropy in X
aniso_z = 20            # anisotropy in Z
angles = 0              # angle of rotation
seed = 20230406         # seed

############################################################################################################

# store the random field models inside layers
layers = generate_rf_group(aniso_x, aniso_z, angles, seed)


##############################################################################################

matrix = np.zeros((z_max, x_max))
coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
values = np.zeros(coords_to_list.shape[0])


y = layer_boundary(x_coord)

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