import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.spatial import Delaunay

from layers_functions.rf_model import random_field_generator, random_field_generator2
from layers_functions.rf_polygons import generate_2D_polygons

##### MAIN VARIABLES ######################################################################################
n_layers = 5            # no. of layers
x_max = 256             # length (x) of the model
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
no_realizations = 1000    # number of realizations to generate

for counter, seed in enumerate(range(seed, seed + no_realizations, 1)):
    print('Generating model no.:', counter)
    random.seed(seed)

    # generate the random polygons within the model
    surfaces = generate_2D_polygons(z_max, n_layers, ndim, x_coord, z_coord)

    # generate the random field models for different materials
    srf_sand = random_field_generator(0.3, 1.5, aniso_x, aniso_z, ndim, seed+1)
    srf_clay = random_field_generator(0.3, 2.1, aniso_x, aniso_z, ndim, seed+2)
    srf_silt = random_field_generator(0.5, 3.2, aniso_x, aniso_z, ndim, seed+3)
    srf_other = random_field_generator2(ndim, var, len_scale, anis, angles, mean, seed+4)
    # store the random field models inside layers
    layers = [srf_sand, srf_silt, srf_clay, srf_sand, srf_clay, srf_silt, srf_sand]
    random.shuffle(layers)  # shuffle the order of the list of materials for each loop

    # transform all coordinates to a single list
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
    # create a list filled with zeros of the same size as all the coordinates
    values = np.zeros(coords_to_list.shape[0])
    # create an empty container for the random layer coordinates
    layer_coordinates = []

    # loop over each pair of consecutive surfaces
    for i, surfaces in enumerate(zip(surfaces, surfaces[1:])):
        # unpack the consecutive surfaces into to variables top and bot
        top_surf, bot_surf = surfaces
        # concatenate the coordinates over a single list
        poly_points = list(top_surf) + list(bot_surf)

        # using Delunay triangulation, check if the coordinates are inside of outside the polygon
        # this will return a mask of boolean values for inside/outside points
        mask = Delaunay(poly_points, qhull_options = "QJ").find_simplex(coords_to_list) >= 0
        # run the list of coordinates through the mask and store the true points
        layer_coordinates = coords_to_list[mask]

        # evaluate the corresponding SRF to the current polygon
        layer_IC = layers[i](layer_coordinates.T)
        # store the results of each mask in the values variable
        values[mask] = layer_IC

    # store the results in a dataframe
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})
    grouped = df.groupby('x')


    ##### PLOT AND SAVE THE RESULTS ########################################################################
    plt.clf()   # clear the current figure
    df_pivot = df.pivot(index="z", columns="x", values="IC")

    fig, ax = plt.subplots(figsize=(2.56, .64))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    plt.savefig(f"cs2d\\cs_{counter}.png")  # save the cross-section
    df.to_csv(f"cs2d\\cs_{counter}.csv")
    plt.close()
