import numpy as np
import pykrige
from scipy.spatial import cKDTree


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

    nb_near_points: int = 5
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


def kriging_interpolation(training_points, training_data, prediction_points):
    # assign to variables
    interpolating_function = pykrige.ok.OrdinaryKriging(
        training_points.T[0],
        training_points.T[1],
        training_data,
        variogram_model='gaussian',
        variogram_parameters={
            "nugget": 40819929,
            "range": 51,
            "sill": 38020807
        },
        verbose=False,
        enable_plotting=False,
        nlags=20,
    )

    zn, ss = interpolating_function.execute(
        "points", prediction_points.T[0], prediction_points.T[1]
    )

    return zn
