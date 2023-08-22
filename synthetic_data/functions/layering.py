import os
import random
import numpy as np
import pandas as pd
from random import betavariate
from functions.generate_rf import generate_rf_group
import matplotlib.pyplot as plt



def layer_boundary(x_coord, z_max):
    """
    Generate a sine or cosine line as a layer boundary.

    Args:
        x_coord (array-like): X coordinates.
        z_max (float): Maximum depth.

    Returns:
        array-like: Y coordinates of the layer boundary.
    """

    x_max = len(x_coord)

    # Generate amplitude using the pert function with specified range
    amplitude = pert(2, 5, z_max)
    # Generate period using the pert function with specified range
    period = pert(x_max, 1000, 10000)
    # Randomly shift the phase of the wave
    phase_shift = np.random.uniform(low=0, high=x_max)
    # Randomly shift the entire wave vertically
    vertical_shift = np.random.uniform(low=0, high=z_max)
    # Choose between sine and cosine wave functions
    func = random.choice([np.sin, np.cos])
    # Generate the y-coordinates using the chosen function and parameters
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y



def pert(low, peak, high, *, lamb=10):
    """
    Generate a value using a Beta-Pert distribution.

    Args:
        low (float): Minimum value.
        peak (float): Most likely value.
        high (float): Maximum value.
        lamb (float, optional): Lambda parameter for the distribution. Defaults to 10.

    Returns:
        float: Generated value.
    """
    r = high - low

    # Calculate alpha and beta parameters for the Beta distribution
    alpha = 1 + lamb * (peak - low) / r
    beta = 1 + lamb * (high - peak) / r
    # Generate a value using the Beta distribution

    return low + betavariate(alpha, beta) * r



def generate_synthetic(output_folder, counter, z_max, x_max, seed):
    """
    Generate synthetic data with given parameters and save results in the specified output folder.

    Parameters:
        output_folder (str): The folder to save the synthetic data.
        counter (int): Current realization number.
        z_max (int): Depth of the model.
        x_max (int): Length of the model.
        seed (int): Seed for random number generation.
    """

    # Define the geometry for the synthetic data generation
    x_coord = np.arange(0, x_max, 1)  # Array of x coordinates
    z_coord = np.arange(0, z_max, 1)  # Array of z coordinates
    xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")  # 2D mesh of coordinates x, z

    # Generate random field models and shuffle them
    layers = generate_rf_group(seed)  # Store the random field models inside layers
    np.random.shuffle(layers)  # Shuffle the layers

    # Set up the matrix geometry
    matrix = np.zeros((z_max, x_max))  # Create the matrix of size {rows, cols}
    coords_to_list = np.array([xs.ravel(), zs.ravel()]).T  # Store the grid coordinates in a variable
    values = np.zeros(coords_to_list.shape[0])  # Create a matrix same as coords but with zeros

    # Generate new y value for each plot and sort them to avoid stacking
    y1 = layer_boundary(x_coord, z_max)
    y2 = layer_boundary(x_coord, z_max)
    y3 = layer_boundary(x_coord, z_max)
    y4 = layer_boundary(x_coord, z_max)
    boundaries = [y1, y2, y3, y4]  # Store the boundaries in a list
    boundaries = sorted(boundaries, key=lambda x: x[0])  # Sort the list to avoid stacking on top of each other

    # Create containers for each layer
    area_1, area_2, area_3, area_4, area_5 = [], [], [], [], []

    # Assign grid cells to each layer based on the boundaries
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

    # Apply the random field models to the layers
    all_layers = [area_1, area_2, area_3, area_4, area_5]
    for i, lst in enumerate(all_layers):
        mask = (coords_to_list[:, None] == all_layers[i]).all(2).any(1)
        layer_coordinates = coords_to_list[mask]
        layer_IC = layers[i](layer_coordinates.T)
        values[mask] = layer_IC

    # Store the results in a dataframe
    df = pd.DataFrame({"x": xs.ravel(), "z": zs.ravel(), "IC": values.ravel()})

    # Plot and save the results
    plt.clf()  # Clear the current figure
    df_pivot = df.pivot(index="z", columns="x", values="IC")
    fig, ax = plt.subplots(figsize=(x_max / 100, z_max / 100))
    ax.set_position([0, 0, 1, 1])
    ax.imshow(df_pivot)
    plt.axis("off")
    filename = f"cs_{counter}"
    fig_path = os.path.join(output_folder, f"{filename}.png")
    csv_path = os.path.join(output_folder, f"{filename}.csv")
    plt.savefig(fig_path)
    df.to_csv(csv_path)
    plt.close()




# from matplotlib import rcParams
#
# # Set the font family to "Arial"
# rcParams['font.family'] = 'Arial'
#
# arr1 = [pert(2, 5, 32) for _ in range(10_000)]
# arr2 = [pert(512, 1000, 10000) for _ in range(10_000)]
# arr3 = np.random.uniform(0, 512, 10_000)
# arr4 = np.random.uniform(0, 32, 10_000)
#
# fig, axs = plt.subplots(2, 2, figsize=(9, 5))
#
# axs[0, 0].hist(arr1, bins=50, alpha=0.5, color='silver', edgecolor='dimgray', density=True)
# axs[0, 0].set_xlabel('(a)  Amplitude', fontsize=10)
# axs[0, 0].set_ylabel('Relative Frequency', fontsize=10)
# axs[0, 0].tick_params(axis='both', labelsize=8)
#
# axs[0, 1].hist(arr2, bins=50, alpha=0.5, color='silver', edgecolor='dimgray', density=True)
# axs[0, 1].set_xlabel('(b)  Period', fontsize=10)
# axs[0, 1].set_ylabel('Relative Frequency', fontsize=10)
# axs[0, 1].tick_params(axis='both', labelsize=8)
#
# axs[1, 0].hist(arr3, bins=50, alpha=0.5, color='silver', edgecolor='dimgray', density=True)
# axs[1, 0].set_xlabel('(c)  Phase Shift', fontsize=10)
# axs[1, 0].set_ylabel('Relative Frequency', fontsize=10)
# axs[1, 0].tick_params(axis='both', labelsize=8)
#
# axs[1, 1].hist(arr4, bins=50, alpha=0.5, color='silver', edgecolor='dimgray', density=True)
# axs[1, 1].set_xlabel('(d)  Vertical Shift', fontsize=10)
# axs[1, 1].set_ylabel('Relative Frequency', fontsize=10)
# axs[1, 1].tick_params(axis='both', labelsize=8)
#
# plt.tight_layout()
# plt.show()