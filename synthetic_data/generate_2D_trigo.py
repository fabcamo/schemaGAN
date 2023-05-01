import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from layers_functions.layer_boundary import layer_boundary
from layers_functions.generate_rf import generate_rf_group
from layers_functions.split_data import split_data

# Check the time and start the timers
time_current = time.strftime("%d/%m/%Y %H:%M:%S")

output_folder = 'C:\\inpt\\synthetic_data\\512x32'

# Generate a random seed using NumPy
seed = np.random.randint(20220412, 20230412)
# Set the seed for NumPy's random number generator
np.random.seed(seed)

no_realizations = 200     # number of realizations to generate
train_size = 0.9


##### MAIN DIMENSION VARIABLES ######################################################################################
x_max = 512             # length (x) of the model
z_max = 32              # depth (z) of the model
# Model coordinates
x_coord = np.arange(0, x_max, 1)       # array of x coordinates
z_coord = np.arange(0, z_max, 1)       # array of z coordinates
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")   # 2D mesh of coordinates x,z
############################################################################################################


time_start = time.time() # start the timer
counter = 0
while counter < no_realizations:
    try:
        print('Generating model no.:', counter+1)
        #random.seed(seed)

        # store the random field models inside layers
        layers = generate_rf_group(seed)
        np.random.shuffle(layers)
        # set up the geometry
        matrix = np.zeros((z_max, x_max))
        coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
        values = np.zeros(coords_to_list.shape[0])

        # Generate new y value for each plot
        y1 = layer_boundary(x_coord, z_max)
        y2 = layer_boundary(x_coord, z_max)
        y3 = layer_boundary(x_coord, z_max)
        y4 = layer_boundary(x_coord, z_max)
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


        ##### PLOT AND SAVE THE RESULTS ########################################################################
        plt.clf()   # clear the current figure
        df_pivot = df.pivot(index="z", columns="x", values="IC")

        fig, ax = plt.subplots(figsize=(x_max/100, z_max/100))
        ax.set_position([0, 0, 1, 1])
        ax.imshow(df_pivot)
        plt.axis("off")
        filename = f"cs_{counter}"
        fig_path = os.path.join(output_folder, f"{filename}.png")
        csv_path = os.path.join(output_folder, f"{filename}.csv")
        plt.savefig(fig_path)
        df.to_csv(csv_path)
        plt.close()

        counter += 1

    except Exception as e:
        print(f"Error in generating model no. {counter + 1}: {e}")
        continue


# Split the data into train and validation
split_data(output_folder, os.path.join(output_folder, "train"),
           os.path.join(output_folder, "./validation"), train_size)

# Once the files are moved to the train and validation folders, delete them from output
file_list = os.listdir(output_folder)   # Get a list of all files in the folder
# Iterate through the files and delete the CSV files
for file_name in file_list:
    if file_name.endswith('.csv'):
        file_path = os.path.join(output_folder, file_name)
        os.remove(file_path)

time_end = time.time() # End the timer
# Execution time of the model
execution_time = abs(time_start - time_end) # Calculate the run time
# Format time taken to run into> Hours : Minutes : Seconds
hours = int(execution_time // 3600)
minutes = int((execution_time % 3600) // 60)
seconds = int(execution_time % 60)
time_str = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

# Save the seed to a text file in the specified path
file_path = os.path.join(output_folder, 'random_seed.txt')
with open(file_path, 'w') as f:
    f.write("Executed on: {}\n".format(time_current))
    f.write("Execution time: {}\n\n".format(time_str))
    f.write("Seed: {}\n\n".format(seed))
    f.write("No. of realizations: {}\n\n".format(no_realizations))

