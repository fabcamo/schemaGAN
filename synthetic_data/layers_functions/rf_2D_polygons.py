import numpy as np
import random


# create a defined number of random Z points for the polygons
def random_depths(z_max, n_layers, ndim):
    z_random = np.array(
        [sorted(random.sample(range(z_max), n_layers-1))
         for i in range(ndim)]  # 2 is the points per line
    )
    z_random = z_random.T
    return z_random


# For any given depth (z), gather the 2D coordinates of the polygon
def poly_points_2D(x_coord, z):
    return [
        (min(x_coord), z[0]),
        (max(x_coord), z[1])
    ]

# create the 2D polygons from the random Z points
def generate_2D_polygons(z_max, n_layers, ndim, x_coord, z_coord):
    points_per_poly = 2
    surfaces = np.zeros((n_layers+1, points_per_poly, ndim))      # create the container filled with zeros

    # Fix the top and bottom polygons
    top_surface = [(min(x_coord), min(z_coord)), (max(x_coord), min(z_coord))]
    bot_surface = [(min(x_coord), max(z_coord)), (max(x_coord), max(z_coord))]
    surfaces[0] = top_surface
    surfaces[-1] = bot_surface

    # create random depth values depending on the layers
    z_random = random_depths(z_max, n_layers, ndim)
    print(z_random)

    # fill the other surfaces with the random depths
    surfaces[1:-1] = [poly_points_2D(x_coord, z) for z in z_random]   # for every random Z, generate a surface

    return surfaces

