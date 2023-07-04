from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv


def read_files(path):
    """
    Read all files in a directory

    Parameters
    ----------
    path: str
        path to directory

    Returns
    -------
    cpt_files: list
    """
    cpts = os.listdir(path)
    cpt_files = [Path(path, c) for c in cpts]
    return cpt_files


def process_cpts(cpts):
    """
    Process all cpt files using the geolib_plus library (Robertson classification with Lengkeek unit weight method)

    Parameters
    ----------
    cpts: list
        list of cpt files

    Returns
    -------
    data: dict
    """
    data = {"IC": [],
            "depth": [],
            "coordinates": [],
            }

    for cpt in cpts:
        # initialize models
        cpt_gef = GefCpt()

        # read the cpt for each type of file
        cpt_gef.read(cpt)

        # do pre-processing
        cpt_gef.pre_process_data()

        # do pre-processing
        interpreter = RobertsonCptInterpretation()
        interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK

        interpreter.user_defined_water_level = True
        cpt_gef.pwp = 0

        cpt_gef.interpret_cpt(interpreter)

        data["IC"].append(cpt_gef.IC)
        data["depth"].append(cpt_gef.depth_to_reference)
        data["coordinates"].append(cpt_gef.coordinates)
    return data

if __name__ == "__main__":
    cpts_g = read_files(r"2422-220749_Sonderingen_GEF")
    data_cpts1 = process_cpts(cpts_g)


    print(data_cpts1['coordinates'])

    plt.plot(np.array(data_cpts1["coordinates"])[:, 0],
             np.array(data_cpts1["coordinates"])[:, 1],
             marker="o", linewidth=0)
    plt.show()
    plt.clf()

    # Save coordinates as CSV
    output_filename = "data_cpts1_coordinates.csv"
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])  # Write column headers
        writer.writerows(data_cpts1["coordinates"])

    print(f"Coordinates saved to '{output_filename}'")

    fig, axes = plt.subplots(1, 30, figsize=(20, 2), sharey=True)  # Create subplots with 1 row and 30 columns
    # Plot each element in its own subplot
    for i in range(30):
        axes[i].plot(data_cpts1["IC"][i], data_cpts1["depth"][i], color='navy', label="IC")
        axes[i].set_title(f"CPT{i + 1}")  # Set a title for each subplot
        axes[i].set_ylim(-35, 0)  # Set the y-axis range
        axes[i].set_xlim(1, 4)  # Set the y-axis range

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

    cpts_g = read_files(r"cpts_BRO_in_embankment")
    data_cpts2 = process_cpts(cpts_g)

    plt.plot(np.array(data_cpts2["coordinates"])[:, 0],
             np.array(data_cpts2["coordinates"])[:, 1],
              marker="o", linewidth=0)
    plt.show()
    plt.clf()

    # Save coordinates as CSV
    output_filename = "data_cpts2_coordinates.csv"
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])  # Write column headers
        writer.writerows(data_cpts2["coordinates"])
    print(f"Coordinates saved to '{output_filename}'")


    fig, axes = plt.subplots(1, 24, figsize=(15, 3), sharey=True)  # Create subplots with 1 row and 30 columns
    # Plot each element in its own subplot
    for i in range(24):
        axes[i].plot(data_cpts2["IC"][i], data_cpts2["depth"][i], color='darkred', label="IC")
        axes[i].set_title(f"CPT{i + 1}")  # Set a title for each subplot
        axes[i].set_ylim(-35, 0)  # Set the y-axis range
        axes[i].set_xlim(1, 4)  # Set the y-axis range

plt.tight_layout()  # Adjust spacing between subplots
plt.show()