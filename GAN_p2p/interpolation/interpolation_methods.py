import numpy as np
from scipy.spatial import cKDTree, Delaunay
from pykrige.ok import OrdinaryKriging
import skgstat as skg
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage.restoration import inpaint


def nearest_interpolation(training_points, training_data, prediction_points):
    """
    Interpolate data using the Nearest Neighbour method.

    Parameters:
        training_points (np.array): Array containing the known data points.
        training_data (np.array): Array containing the known data values.
        prediction_points (np.array): Array containing the points for which predictions are to be made.

    Returns:
        np.array: An array containing the interpolated data.
    """

    # Create a KDtree from the known data points
    # KDtree is used to quickly find nearest neighbours in multidimensional space
    tree = cKDTree(training_points)

    # Query the KDtree to find the closest data point for each prediction point
    # 'dist' is the distance to the closest point, 'idx' is the index of the closest point in the training data
    dist, idx = tree.query(prediction_points)

    # Initialize an empty list to store the interpolated data
    zn = []

    # Iterate through each prediction point
    for i in range(len(prediction_points)):
        # Add the value of the closest training data to the list
        zn.append(training_data[idx[i]])

    # Convert the list to a numpy array
    zn = np.array(zn)

    return zn



def idw_interpolation(training_points, training_data, prediction_points):
    """
    Performs Inverse Distance Weighting (IDW) interpolation on the provided data.

    Parameters:
        training_points (np.ndarray): Known data points (x, y coordinates).
        training_data (np.ndarray): Known data values corresponding to the points.
        prediction_points (np.ndarray): Points where to interpolate.

    Returns:
        np.ndarray: Interpolated values at each point in the prediction_points.
    """


    # Power parameter for the IDW
    power = 1.0

    # Small value added to distances to prevent division by zero
    tol = 1e-9

    # Convert inputs to numpy arrays
    training_points = np.array(training_points)
    training_data = np.array(training_data)

    # Get unique x coordinates from the training points
    x_points = np.array(list(set(training_points[:, 1]))).reshape(-1, 1)

    # Number of nearest points to consider for the interpolation
    nb_near_points = len(x_points)
    # Get training data for the points
    training_data_points = []
    for x in x_points:
        idx = np.where(training_points[:, 1] == x)[0]
        training_data_points.append(training_data[idx])
    training_data_points = np.array(training_data_points)

    # Get unique x coordinates from the prediction points
    x_points_prediction = np.array(list(set(prediction_points[:, 1]))).reshape(-1, 1)
    y_points_prediction = np.array(list(set(prediction_points[:, 0])))


    # Compute Euclidean distances from prediction_points to training_points
    tree = cKDTree(x_points)

    # Get distances and indexes of the nb_near_points closest training points
    dist, idx = tree.query(x_points_prediction, nb_near_points)

    # Add small value to distances to prevent division by zero
    dist += tol

    # Initialize list to store interpolated values
    zn = []

    # Loop through each prediction point
    for i in range(len(x_points_prediction)):
        # Retrieve the training data corresponding to the closest points
        data = training_data_points[idx[i]]
        # extrapolate data
        data = extrapolate_array(data, len(y_points_prediction))

        # Compute the IDW interpolation for the current point
        zn.append(np.sum(data.T / dist[i] ** power, axis=1) / np.sum(1.0 / dist[i] ** power))

    # Convert interpolated values list to a numpy array
    zn = np.array(zn)

    return zn



def kriging_interpolation(training_points, training_data, gridx, gridy):
    """Perform ordinary kriging interpolation.

    Parameters:
        training_points (np.ndarray): Known data points, each row is a pair (y, x).
        training_data (np.ndarray): Known data values corresponding to the points.
        gridx (np.ndarray): Grid points in x dimension for performing interpolation.
        gridy (np.ndarray): Grid points in y dimension for performing interpolation.

    Returns:
        np.ndarray: Interpolated values at each point in the grid.
    """

    # Separate the training points into rows and columns coordinates
    # Rows -> y-axis
    y = training_points[:, 0]
    # Columns -> x-axis
    x = training_points[:, 1]

    X = np.column_stack((x, y))

    # The training data are the z values at each (x, y) coordinate
    # z = training_data

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    # Create the Gaussian process regressor
    kernel = RBF(length_scale=(50, 0.5)) + WhiteKernel()  # You can adjust the length_scale parameter as needed
    regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

    # Fit the regressor to the data
    regressor.fit(X, training_data)
    print(f"Kernel parameters: {regressor.kernel_}")

    # Flatten the meshgrid for prediction
    XX, YY = np.meshgrid(gridx, gridy)
    X_interpolated = np.column_stack((XX.ravel(), YY.ravel()))

    # Predict pixel values for the interpolated coordinates
    y_interpolated = regressor.predict(X_interpolated)

    # Reshape the predicted values into the shape of the interpolated image
    interpolated_image = np.reshape(y_interpolated, (len(gridx), len(gridy)))

    return interpolated_image




def natural_nei_interpolation(training_points, training_data, prediction_points):
    """
    Interpolate data using the Natural Neighbour method.

    Parameters:
        training_points (np.array): Array containing the known data points.
        training_data (np.array): Array containing the known data values.
        prediction_points (np.array): Array containing the points for which predictions are to be made.

    Returns:
        list: A list containing the interpolated data.
    """

    # Initialize an empty list to store the interpolated data
    zn = []

    # define natural neighbor interpolation
    natural_n = interpolate.LinearNDInterpolator(np.array(training_points), np.array(training_data),
                                                    fill_value=np.mean(np.array(training_data)))

    # create interpolation for every point
    for i in range(len(prediction_points)):
        zn.append(natural_n(prediction_points[i]))

    return np.array(zn)

def inpt_interpolation(training_points, training_data, prediction_points, image_size=(32, 512)):

    # create mask field
    mask = np.ones(len(prediction_points))
    # create defected image field
    image = np.zeros(mask.shape)

    # get the closest to the image
    tree = cKDTree(prediction_points)
    _, idx = tree.query(training_points, 1)
    # create image
    image[idx] = training_data
    # mask with know pixels need to be zero
    mask[idx] = 0

    # plt.imshow(image.reshape(image_size))
    # plt.show()

    # inpaint
    zn = inpaint.inpaint_biharmonic(image.reshape(image_size), mask.reshape(image_size))
    return zn



def extrapolate_array(arr, max_length):
    # Create an empty array with the desired length
    extrapolated_arr = np.empty((len(arr), max_length))

    # Perform linear interpolation for each array in the input
    for i, a in enumerate(arr):
        x = np.linspace(0, 1, len(a))
        x_new = np.linspace(0, 1, max_length)
        extrapolated_arr[i] = np.interp(x_new, x, a)
    return extrapolated_arr



# Old Methods
#####################################################################################################################










def natural_nei_interpolation_old(training_points, training_data, prediction_points):
    """
    Interpolate data using the Natural Neighbour method.

    Parameters:
        training_points (np.array): Array containing the known data points.
        training_data (np.array): Array containing the known data values.
        prediction_points (np.array): Array containing the points for which predictions are to be made.

    Returns:
        list: A list containing the interpolated data.
    """

    # Initialize an empty list to store the interpolated data
    zn = []

    # Convert prediction points to numpy array
    prediction_points = np.array(prediction_points)

    # Iterate through each prediction point
    for i in range(len(prediction_points)):
        # Add the current prediction point to the known data points
        new_points = np.vstack([training_points, prediction_points[i].T])

        # Create Delaunay triangulation from the data points
        tri = Delaunay(new_points)

        # Find index of current prediction point in the Delaunay triangulation
        pindex = np.where(np.all(tri.points == prediction_points[i].T, axis=1))[0][0]

        # Find indices of neighbour points in the Delaunay triangulation
        neig_idx = tri.vertex_neighbor_vertices[1][
            tri.vertex_neighbor_vertices[0][pindex]: tri.vertex_neighbor_vertices[0][pindex + 1]
        ]

        # Get coordinates of the neighbour points
        coords_neig = [tri.points[j] for j in neig_idx]

        # Compute Euclidean distance from current prediction point to each neighbour point
        dist = [np.linalg.norm(prediction_points[i] - j) for j in coords_neig]

        # Find index of each neighbour point in the training data
        idx_coords_neig = [np.where(np.all(training_points == j, axis=1))[0][0] for j in coords_neig]

        # Get data values at the neighbour points
        data_neig = [training_data[j] for j in idx_coords_neig]

        # Compute weights and apply them to the data values to get the interpolated value
        zn_aux = [data_neig[ii] * dist[ii] / np.sum(dist) for ii in range(len(data_neig))]

        # Sum up the weighted data values and append to the list
        zn.append(np.sum(zn_aux, axis=0))

    return zn





def kriging_interpolation_old(training_points, training_data, gridx, gridy):
    """
    Perform ordinary kriging interpolation.

    Parameters:
        training_points (np.ndarray): Known data points, each row is a pair (y, x).
        training_data (np.ndarray): Known data values corresponding to the points.
        gridx (np.ndarray): Grid points in x dimension for performing interpolation.
        gridy (np.ndarray): Grid points in y dimension for performing interpolation.

    Returns:
        np.ndarray: Interpolated values at each point in the grid.
    """

    # Separate the training points into rows and columns coordinates
    # Rows -> y-axis
    y = training_points[:, 0]
    # Columns -> x-axis
    x = training_points[:, 1]

    # The training data are the z values at each (x, y) coordinate
    z = training_data

    # Define the variogram model to be used
    variogram_model = 'gaussian'

    # Estimate variogram model parameters using scikit-gstat
    coordinates = np.column_stack((x, y))
    V = skg.Variogram(coordinates, z, model=variogram_model, maxlag=0.5, n_lags=10)

    # Fit the variogram model
    V.fit()

    # Plot the variogram
    V.plot()

    # Get estimated variogram parameters: sill, range and nugget
    sill = V.parameters[0]
    range = V.parameters[1]
    nugget = V.parameters[2]

    # Define the parameters for the ordinary kriging
    nlags = 6
    weight = False
    anisotropy_scaling = 1.0
    anisotropy_angle = 0.0
    variogram_parameters = {'sill': sill, 'range': range, 'nugget': nugget}
    verbose = False
    enable_plotting = False

    # Create ordinary kriging object
    OK = OrdinaryKriging(
        x, y, z,
        variogram_model=variogram_model,
        nlags=nlags,
        weight=weight,
        anisotropy_scaling=anisotropy_scaling,
        anisotropy_angle=anisotropy_angle,
        variogram_parameters=variogram_parameters,
        verbose=verbose,
        enable_plotting=enable_plotting
    )

    # Execute on grid
    zn, _ = OK.execute('grid', gridx, gridy)

    return zn
