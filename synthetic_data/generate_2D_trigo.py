import numpy as np
from datetime import datetime
import random
import pandas as pd
import matplotlib.pyplot as plt

from layers_functions.layer_boundary import layer_boundary
from layers_functions.generate_rf import generate_rf_group

seed = 20230406         # seed
no_realizations = 10     # number of realizations to generate

##### MAIN DIMENSION VARIABLES ######################################################################################
x_max = 512             # length (x) of the model
z_max = 64              # depth (z) of the model
# Model coordinates
x_coord = np.arange(0, x_max, 1)       # array of x coordinates
z_coord = np.arange(0, z_max, 1)       # array of z coordinates
xs, zs = np.meshgrid(x_coord, z_coord, indexing="ij")   # 2D mesh of coordinates x,z

############################################################################################################
start1 = datetime.now()
counter = 0
while counter < no_realizations:
    try:
        print('Generating model no.:', counter+1)
        #random.seed(seed)

        # store the random field models inside layers
        layers = generate_rf_group(seed)
        random.shuffle(layers)
        # set up the geometry
        matrix = np.zeros((z_max, x_max))
        coords_to_list = np.array([xs.ravel(), zs.ravel()]).T
        values = np.zeros(coords_to_list.shape[0])

        # Generate new y value for each plot
        y1 = layer_boundary(x_coord)
        y2 = layer_boundary(x_coord)
        y3 = layer_boundary(x_coord)
        y4 = layer_boundary(x_coord)
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
        plt.savefig(f"test\\cs_{counter}.png")  # save the cross-section
        df.to_csv(f"test\\cs_{counter}.csv")
        plt.close()

        counter += 1

    except Exception as e:
        print(f"Error in generating model no. {counter + 1}: {e}")
        continue


stop1 = datetime.now()
# Execution time of the model
execution_time = stop1 - start1
print("Execution time is: ", execution_time)
