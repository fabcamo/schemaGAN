import numpy as np
import matplotlib.pyplot as plt
import gstools as gs
import random
from matplotlib.patches import Polygon          # Import Polygon class from matplotlib's patches module
from scipy.spatial import ConvexHull, Delaunay
import plotly.graph_objects as go
import pandas as pd

from layers_functions.rf_model import random_field_generator


# sample n number of depth values to generate the random layer lines
def sample_depths(height, n_layers, ndim):
    y_rand = np.array(
        [sorted(random.sample(range(height), n_layers-1))
         for i in range(ndim)]  # 2 is the points per line
    )
    # y_rand = y_rand.T
    return y_rand


def surface_points_from_depth(x_coord, y):
    line_rand = [(min(x_coord), y[0]), (max(x_coord), y[1])]
    return line_rand


# Create an array full of pair of points for all the lines in the 2D random field
def generate_all_lines(n_layers,points_per_line,ndim):
    # Create an array filled with zeros with the required no. lines and coordinates per line
    points_for_lines = np.zeros(
        (n_layers + 1, points_per_line, ndim))  # (no. lines+1, data_per_line, no. of points to define the line)
    # Fix the top and bottom line as the MAX and MIN coordinates
    line_top = [(min(x_coord), max(y_coord)), (max(x_coord), max(y_coord))]
    line_bot = [(min(x_coord), min(y_coord)), (max(x_coord), min(y_coord))]
    # Replace the top and bottom lines with their actual values
    points_for_lines[0] = line_top
    points_for_lines[-1] = line_bot
    # Call the random depths
    y_rand = sample_depths(height, n_layers, ndim)

    points_for_lines[1:-1] = [
        surface_points_from_depth(x_coord, y) for y in y_rand
    ]
    #print(points_for_lines)
    return points_for_lines


def generate_polygons(n_layers,points_per_line,ndim):
    polygons = []
    points_for_lines = generate_all_lines(n_layers,points_per_line,ndim)

    for i in range(n_layers):
        points = [points_for_lines[i], points_for_lines[i + 1][::-1]]
        points = np.concatenate(points).reshape(4, 2)
        polygons.append(points)

    return polygons



##### VARIABLES ########################################################################################################
ndim = 2            # number of dimensions
var = 0.8           # variance
len_scale = 32      # main length scale
anis = 1 / 2        # anisotropy
angles = 0          # angle of rotation
mean = 2            # mean
# Define the geometry of the random field
width = 256         # width of the model
height = 64         # height of the model
# Create the coordinates of the space
x_coord = np.linspace(0, width, width+1)
y_coord = np.linspace(0, height, height+1)
# Create the mesh
xs, ys = np.meshgrid(x_coord, y_coord, indexing="ij")

# number of layers
n_layers = 3
points_per_line = 2
########################################################################################################################

polygons = generate_polygons(n_layers,points_per_line,ndim)

xx, yy = np.meshgrid(x_coord, y_coord)                      # Create a 2D grid of points from the x and y arrays
mesh = np.vstack((xx.flatten(), yy.flatten())).T # Flatten and stack the 2D grid of points into a 2D array of coordinates

# Group points based on polygons
groups = []                                     # Create an empty list to store the groups
for p in polygons:                              # Loop over each polygon
    path = Polygon(p).get_path()                # Create a Path object from the polygon's points
    mask = path.contains_points(mesh)           # Generate a boolean mask indicating whether each point in the mesh is inside or outside the polygon
    groups.append(mask)                         # Append the mask to the groups list

# Get coordinates of points inside polygon1
mask1 = groups[0]
x1, y1 = mesh[np.where(mask1)].T

# Get coordinates of points inside polygon2
mask2 = groups[1]
x2, y2 = mesh[np.where(mask2)].T

# Get coordinates of points inside polygon2
mask3 = groups[2]
x3, y3 = mesh[np.where(mask3)].T


# Plot the results
plt.figure(figsize=(6,6))                       # Create a new figure with a 6x6 inch size
plt.scatter(mesh[:,0], mesh[:,1], c=groups[0].astype(int) + groups[1].astype(int)*2 + groups[2].astype(int)*3) # Create a scatter plot of the points on the mesh, with colors determined by the groups
plt.gca().set_aspect('equal')                   # Set the aspect ratio to equal
plt.show()                                      # Display the plot















srf_test = random_field_generator(ndim, var, len_scale, anis, angles, mean)
# Evaluate the randon field model on the coordinates
field = srf_test((x1, y1), mesh_type='structured')



ax = srf_test.plot()
ax.set_aspect("equal")

plt.show()
