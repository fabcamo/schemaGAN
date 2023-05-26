import numpy as np
import pykrige
from scipy.spatial import cKDTree, Delaunay
from pykrige.ok import OrdinaryKriging


# From DataFusionTools> nearest neighbor interpolation
def nearest_interpolation(training_points, training_data, prediction_points):

    # create KDtree
    tree = cKDTree(training_points)

    # compute closest distance and index of the closest index
    dist, idx = tree.query(prediction_points)
    zn = []
    # create interpolation for every point
    for i in range(len(prediction_points)):
        # interpolate
        zn.append(training_data[idx[i]])

    zn = np.array(zn)

    return zn


def idw_interpolation(training_points, training_data, prediction_points):

    nb_near_points: int = 6
    power: float = 1.0
    tol: float = 1e-9

    # assign to variables
    training_points = np.array(training_points)  # training points
    training_data = np.array(training_data)  # data at the training points

    # compute Euclidean distance from grid to training
    tree = cKDTree(training_points)

    # get distances and indexes of the closest nb_points
    dist, idx = tree.query(prediction_points, nb_near_points)
    dist += tol  # to overcome division by zero
    zn = []

    # create interpolation for every point
    for i in range(len(prediction_points)):
        # compute weights
        data = training_data[idx[i]]

        # interpolate
        zn.append(
            np.sum(data.T / dist[i] ** power)
            / np.sum(1.0 / dist[i] ** power)
        )

    zn = np.array(zn)

    return zn


def kriging_interpolation(training_points, training_data, gridx, gridy):
    # assign to variables
    variogram_model = 'gaussian'
    nlags = 6

    x = training_points[:, 1] # take the columns
    y = training_points[:, 0] # take the rows
    z = training_data

    # Create the kriging object
    OK = OrdinaryKriging(x, y, z, variogram_model=variogram_model, nlags=nlags)
    # Interpolate the data onto the grid
    z_interp, _ = OK.execute('grid', gridx, gridy)

    return z_interp




def natural_nei_interpolation(training_points, training_data, prediction_points):

    zn = []
    prediction_points = np.array(prediction_points)
    for i in range(len(prediction_points)):
        new_points = np.vstack([training_points, prediction_points[i].T])
        tri = Delaunay(new_points)
        # Find index of prediction point
        pindex = np.where(np.all(tri.points == prediction_points[i].T, axis=1))[0][
            0
        ]
        # find neighbours
        neig_idx = tri.vertex_neighbor_vertices[1][
                   tri.vertex_neighbor_vertices[0][pindex]: tri.vertex_neighbor_vertices[
                       0
                   ][pindex + 1]
                   ]

        # get the coordinates of the neighbours
        coords_neig = [tri.points[j] for j in neig_idx]
        # compute Euclidean distance
        dist = [np.linalg.norm(prediction_points[i] - j) for j in coords_neig]
        # find data of the neighbours
        idx_coords_neig = []
        for j in coords_neig:
            idx_coords_neig.append(
                np.where(np.all(training_points == j, axis=1))[0][0]
            )

        # get data of the neighbours
        data_neig = [np.array(training_data[j]) for j in idx_coords_neig]
        # compute weights
        zn_aux = []
        for ii in range(len(data_neig)):
            aux = data_neig[ii] * dist[ii] / np.sum(dist)
            zn_aux.append(aux)
        zn.append(np.sum(np.array(zn_aux), axis=0))


    return zn
