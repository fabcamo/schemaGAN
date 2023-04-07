import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from layers_functions.layer_boundary import layer_boundary
from layers_functions.generate_rf import generate_rf_group


x_max = 512             # length (x) of the model
z_max = 64              # depth (z) of the model
# Model coordinates
x_coord = np.arange(0, x_max, 1)       # array of x coordinates
z_coord = np.arange(0, z_max, 1)       # array of z coordinates
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")   # 2D mesh of coordinates x,z


##### RANDOM FIELD PARAMETERS ##############################################################################
std_value = 0.3         # standard deviation value
mean = 2.1              # mean
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

# Create mask to split matrix into two layers
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