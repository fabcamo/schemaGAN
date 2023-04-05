import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_max = 512             # length (x) of the model
z_max = 64              # depth (z) of the model
# Model coordinates
x_coord = np.arange(0, x_max, 1)       # array of x coordinates
z_coord = np.arange(0, z_max, 1)       # array of z coordinates
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")   # 2D mesh of coordinates x,z

matrix = np.zeros((z_max, x_max))
print('matrix shape 0>', matrix.shape[0])
print('matrix shape 1>', matrix.shape[1])
coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
print('coords to list')
print(coords_to_list)
values = np.zeros(coords_to_list.shape[0])

amplitude = 60
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
            above_list.append([col,row])
        else:
            below_list.append([col,row])



# Store lists in a list of lists
above_array = np.array(above_list)
print('above array')
print(above_array)
below_array = np.array(below_list)
print('below array')
print(below_array)
lists = [above_array, below_array]


a = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 1], [1, 2], [1, 3], [1, 4]])
b = np.array([[0, 2], [1, 1], [1, 4]])

a = coords_to_list
b = above_array



mask = (a[:,None] == b).all(2).any(1)
print (mask)
print(np.count_nonzero(mask))