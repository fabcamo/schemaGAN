import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy import interpolate
from skimage.restoration import inpaint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


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
    Performs Inverse Distance Weighting (IDW) interpol_compare on the provided data.

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

    # Number of nearest points to consider for the interpol_compare
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

        # Compute the IDW interpol_compare for the current point
        zn.append(np.sum(data.T / dist[i] ** power, axis=1) / np.sum(1.0 / dist[i] ** power))

    # Convert interpolated values list to a numpy array
    zn = np.array(zn)

    return zn



def kriging_interpolation(training_points, training_data, gridx, gridy):
    """Perform ordinary kriging interpol_compare.

    Parameters:
        training_points (np.ndarray): Known data points, each row is a pair (y, x).
        training_data (np.ndarray): Known data values corresponding to the points.
        gridx (np.ndarray): Grid points in x dimension for performing interpol_compare.
        gridy (np.ndarray): Grid points in y dimension for performing interpol_compare.

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

    # Create the Gaussian process regressor
    kernel = RBF(length_scale=(50, 0.5), length_scale_bounds=(1e-2, 1e3)) + \
             WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1e1))  # You can adjust the length_scale parameter as needed
    regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True)

    # Fit the regressor to the data
    regressor.fit(X, training_data)
    #print(f"Kernel parameters: {regressor.kernel_}")

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

    # define natural neighbor interpol_compare
    natural_n = interpolate.LinearNDInterpolator(np.array(training_points), np.array(training_data),
                                                    fill_value=np.mean(np.array(training_data)))

    # create interpol_compare for every point
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

    # Perform linear interpol_compare for each array in the input
    for i, a in enumerate(arr):
        x = np.linspace(0, 1, len(a))
        x_new = np.linspace(0, 1, max_length)
        extrapolated_arr[i] = np.interp(x_new, x, a)
    return extrapolated_arr


