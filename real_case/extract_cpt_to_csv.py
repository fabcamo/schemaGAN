from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def read_files(path):
    """
    Read all .gef files in a directory.

    Parameters
    ----------
    path: str
        Path to the directory containing the files.

    Returns
    -------
    cpt_files: list
        List of file paths to .gef files.
    """
    return [Path(path, f) for f in os.listdir(path) if f.endswith('.gef')]


def process_cpts(cpts):
    """
    Process CPT files using geolib_plus library.

    Parameters
    ----------
    cpts: list
        List of CPT file paths.

    Returns
    -------
    data: list
        List of dictionaries with processed CPT data.
    """
    data = []

    for cpt in cpts:
        cpt_gef = GefCpt()
        try:
            cpt_gef.read(cpt)
        except Exception as e:
            print(f"Error reading CPT file {cpt}: {e}")
            continue

        cpt_gef.pre_process_data()

        interpreter = RobertsonCptInterpretation()
        interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK
        interpreter.user_defined_water_level = True
        cpt_gef.pwp = 0

        cpt_gef.interpret_cpt(interpreter)

        # Extract relevant part of the file name
        file_name = cpt.stem  # Get the file name without extension
        cpt_id = file_name.split('_')[0]  # Extract up to the first underscore

        data.append({
            "Name": cpt_id,
            "depth": cpt_gef.depth_to_reference,
            "depth_max": max(cpt_gef.depth_to_reference, default=0),
            "depth_min": min(cpt_gef.depth_to_reference, default=0),
            "IC": cpt_gef.IC,
            "coordinates": cpt_gef.coordinates
        })

    return data


def equalize_top(data_cpts):
    """
    Equalize the starting depth of all CPTs by removing IC and depth data above the lowest maximum depth.

    Parameters
    ----------
    data_cpts: list
        List of dictionaries containing CPT data.

    Returns
    -------
    equalized_cpts: list
        List of dictionaries with adjusted depth and IC data.
    """
    # Make a copy of the original data to keep it unchanged
    equalized_cpts = []

    # Find the lowest maximum depth across all CPTs
    lowest_max_depth = min(cpt['depth_max'] for cpt in data_cpts)
    # Print a message for the depth used for equalization
    print(f"Equalizing to the lowest maximum depth of {lowest_max_depth} m")

    # Equalize the depth and IC data for each CPT
    for cpt in data_cpts:
        # Create a new dictionary to store the equalized data
        equalized_cpt = cpt.copy()

        depth = equalized_cpt['depth']
        IC = equalized_cpt['IC']

        # Filter depth and IC values that are below the lowest maximum depth (strictly below)
        filtered_data = [(d, ic) for d, ic in zip(depth, IC) if d < lowest_max_depth]  # Adjusted to < instead of <=

        # Separate filtered depth and IC values
        equalized_cpt['depth'], equalized_cpt['IC'] = zip(*filtered_data) if filtered_data else ([], [])

        # Update depth_min and depth_max for the equalized data
        equalized_cpt['depth_min'] = min(equalized_cpt['depth'], default=0)
        equalized_cpt['depth_max'] = max(equalized_cpt['depth'], default=0)

        # Append the equalized data to the new list
        equalized_cpts.append(equalized_cpt)

    return equalized_cpts


def equalize_depth(data_cpts, lowest_min_depth):
    """
    Equalize the depth of all CPTs by extending their depths to the lowest minimum depth.
    New depth values are added starting from the current min depth.

    Parameters
    ----------
    data_cpts: list
        List of dictionaries containing CPT data.

    lowest_min_depth: float
        The target depth that all CPTs should match.

    Returns
    -------
    equalized_depth_cpts: list
        List of dictionaries with equalized depths and IC values.
    """
    equalized_depth_cpts = []

    for cpt in data_cpts:
        # Copy the original CPT data
        equalized_cpt = cpt.copy()

        # Get the depth and IC values
        depth = list(equalized_cpt['depth'])
        IC = list(equalized_cpt['IC'])

        # Calculate the depth interval (assuming uniform intervals for each CPT)
        if len(depth) > 1:
            depth_interval = depth[1] - depth[0]
        else:
            # If only one depth value exists, assume an arbitrary interval (set to 1 for now)
            depth_interval = 1

        # Get the current minimum depth
        current_min_depth = min(depth)

        # Calculate how many new depth values need to be added
        num_steps_to_add = int((lowest_min_depth - current_min_depth) / depth_interval)

        # Add new depth values starting from the current minimum depth
        new_depths = [current_min_depth + (i + 1) * depth_interval for i in range(num_steps_to_add)]
        new_ics = [0] * num_steps_to_add  # Add zeros for IC values at the new depths

        # Append the new data to the CPT
        equalized_cpt['depth'] = depth + new_depths
        equalized_cpt['IC'] = IC + new_ics

        # Update the depth_min and depth_max for the equalized data
        equalized_cpt['depth_min'] = min(equalized_cpt['depth'], default=0)
        equalized_cpt['depth_max'] = max(equalized_cpt['depth'], default=0)

        # Append the equalized CPT data to the new list
        equalized_depth_cpts.append(equalized_cpt)

    return equalized_depth_cpts


#TODO: Compress to 32 pixels
def compress_to_32px(equalized_cpts, method="mean"):
    """
    Compress CPT data to 32 pixels by dividing the depth into 32 equal groups
    and aggregating IC values for each group.

    Parameters
    ----------
    equalized_cpts: list
        List of dictionaries containing CPT data with keys 'depth' and 'IC'.

    method: str
        Aggregation method for compression ('mean' or 'max').

    Returns
    -------
    compressed_cpts: list
        List of dictionaries with compressed CPT data (32 depth and IC values).
    """
    if method not in ["mean", "max"]:
        raise ValueError("Invalid method. Use 'mean' or 'max'.")

    compressed_cpts = []

    for cpt in equalized_cpts:
        # Copy the CPT data to avoid modifying the original
        compressed_cpt = {key: cpt[key] for key in cpt if key not in ['depth', 'IC']}

        depth = np.array(cpt['depth'])
        IC = np.array(cpt['IC'])

        # Ensure data is sorted by depth
        sort_indices = np.argsort(depth)
        depth = depth[sort_indices]
        IC = IC[sort_indices]

        # Define 32 equal depth intervals
        depth_bins = np.linspace(0, 31, 33)  # 33 edges for 32 bins

        # Aggregate IC values within each depth interval
        IC_compressed = []
        depth_compressed = []
        for i in range(len(depth_bins) - 1):
            # Find indices of original depths that map into the current bin range (scaled to 0-31)
            depth_min = depth[0]
            depth_max = depth[-1]
            scaled_bins_min = depth_min + (depth_bins[i] / 31) * (depth_max - depth_min)
            scaled_bins_max = depth_min + (depth_bins[i + 1] / 31) * (depth_max - depth_min)

            mask = (depth >= scaled_bins_min) & (depth < scaled_bins_max)

            # Compute aggregated IC for the current bin
            if mask.any():
                if method == "mean":
                    IC_compressed.append(IC[mask].mean())
                elif method == "max":
                    IC_compressed.append(IC[mask].max())
            else:
                IC_compressed.append(0)
            depth_compressed.append(depth_bins[i])  # Assign bin index as depth

        # Store the compressed data
        compressed_cpt['depth'] = depth_compressed
        compressed_cpt['IC'] = IC_compressed
        compressed_cpts.append(compressed_cpt)

    return compressed_cpts


def save_cpt_to_csv(data_cpts: list, output_dir: str):
    """
    Takes a list of CPT data dictionaries and saves them to a single csv file where there is a single column
    of depth (because all depth is the same 0 to 31) and a column for each CPT, using the CPT name as the column header
    and storing the IC value as for each depth value.

    Args:
        data_cpts (list): List of dictionaries containing CPT data after processing.
        output_dir (str): Directory where the CSV file will be saved.

    Returns:
        None
    """
    # Create a DataFrame to store the CPT data
    df = pd.DataFrame(columns=[cpt['Name'] for cpt in data_cpts])

    # Create a depth column with values from 0 to 31
    df['Depth'] = np.arange(32)

    # Add IC values for each CPT to the DataFrame
    for cpt in data_cpts:
        df[cpt['Name']] = np.interp(df['Depth'], cpt['depth'], cpt['IC'], left=0, right=0)

    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, "32px_cptdata.csv")
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


def plot_equalized_depth_cpts(data_cpts_original, data_cpts_modified, data_cpts_32px, num_to_plot=10, lowest_min_depth=0, lowest_max_depth=0):
    """
    Plot individual CPTs before and after equalization in a 3-row, 10-column layout.
    Depth is plotted on the y-axis, IC on the x-axis, with a dotted line indicating the lowest min depth and max depth.

    Parameters
    ----------
    data_cpts_original: list
        List of CPT data dictionaries before equalization.
    data_cpts_modified: list
        List of CPT data dictionaries after equalizing to the lowest max depth.
    data_cpts_32px: list
        List of CPT data dictionaries after equalizing to the lowest min depth.
    num_to_plot: int
        Number of CPTs to plot (default is 10).
    """
    # Limit to the specified number of CPTs
    data_cpts_original = data_cpts_original[:num_to_plot]
    data_cpts_modified = data_cpts_modified[:num_to_plot]
    data_cpts_32px = data_cpts_32px[:num_to_plot]

    fig, axs = plt.subplots(3, num_to_plot, figsize=(12, 9), sharex=True, sharey=True)
    fig.suptitle("CPT Data Before and After Depth Equalization", fontsize=16)

    for i in range(num_to_plot):
        # Plot individual CPT in the top row (before equalization)
        axs[0, i].plot(data_cpts_original[i]['IC'], data_cpts_original[i]['depth'], label="Before Equalized Top")
        axs[0, i].axhline(lowest_max_depth, color='r', linestyle='dotted', label="Lowest Max Depth")
        axs[0, i].axhline(lowest_min_depth, color='r', linestyle='dotted', label="Lowest Max Depth")
        axs[0, i].invert_yaxis()  # Depth increases downward
        axs[0, i].set_title(f"CPT-{i + 1}")
        axs[0, i].tick_params(axis='x', labelsize=8)
        axs[0, i].tick_params(axis='y', labelsize=8)

        # Plot individual CPT in the middle row (after equalized top)
        axs[1, i].plot(data_cpts_modified[i]['IC'], data_cpts_modified[i]['depth'], label="Equalized Top")
        axs[1, i].axhline(lowest_max_depth, color='r', linestyle='dotted', label="Lowest Max Depth")
        axs[1, i].axhline(lowest_min_depth, color='r', linestyle='dotted', label="Lowest Max Depth")
        axs[1, i].invert_yaxis()  # Depth increases downward
        axs[1, i].tick_params(axis='x', labelsize=8)
        axs[1, i].tick_params(axis='y', labelsize=8)

        # Plot individual CPT in the bottom row (after depth equalization to lowest_min_depth)
        axs[2, i].plot(data_cpts_32px[i]['IC'], data_cpts_32px[i]['depth'], label="Equalized Bottom")
        #axs[2, i].invert_yaxis()  # Depth increases downward
        axs[2, i].tick_params(axis='x', labelsize=8)
        axs[2, i].tick_params(axis='y', labelsize=8)

    # Add labels for rows
    axs[0, 0].set_ylabel("Depth (Before)", fontsize=12)
    axs[1, 0].set_ylabel("Depth (Equalized Top)", fontsize=12)
    axs[2, 0].set_ylabel("Depth (Equalized Bottom)", fontsize=12)

    # Set common x-axis label
    fig.supxlabel("IC", fontsize=14)

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.close()


def plot_compression_results(equalized_cpts, compressed_cpts, num_to_plot=10):
    """
    Plot the results of compression for comparison.

    Parameters
    ----------
    equalized_cpts: list
        List of dictionaries containing original equalized CPT data.

    compressed_cpts: list
        List of dictionaries containing compressed CPT data.

    num_to_plot: int
        Number of CPTs to plot (default is 10).
    """
    fig, axs = plt.subplots(1, num_to_plot, figsize=(num_to_plot * 4, 6), sharey=True)

    for i in range(num_to_plot):
        eq_cpt = equalized_cpts[i]
        comp_cpt = compressed_cpts[i]

        ax = axs[i] if num_to_plot > 1 else axs

        # Plot equalized CPT
        ax.plot(eq_cpt['IC'], eq_cpt['depth'], label='Equalized', color='blue')

        # Plot compressed CPT with a secondary y-axis
        ax_twin = ax.twinx()
        ax_twin.plot(comp_cpt['IC'], comp_cpt['depth'], label='Compressed', color='red', linestyle='--')

        # Formatting
        ax.invert_yaxis()  # Depth increases downward
        ax.set_title(f"CPT-{eq_cpt['Name']}", fontsize=10)
        if i == 0:
            ax.set_ylabel("Depth (m)")

        ax.set_xlabel("IC (Equalized)")
        ax_twin.set_ylabel("Depth (Compressed)")

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Directory containing the CPT files
    cpts_path = read_files(r"D:\schemaGAN\data\groningen")

    # Process CPT files
    data_cpts = process_cpts(cpts_path)

    # Create a copy of the original data for plotting
    original_data_cpts = [cpt.copy() for cpt in data_cpts]

    # Find the lowest maximum and minimum depth across all CPTs
    lowest_max_depth = min(cpt['depth_max'] for cpt in data_cpts)
    lowest_min_depth = min(cpt['depth_min'] for cpt in data_cpts)

    # Equalize depths to match the lowest_max_depth (equalized top)
    equalized_top_cpts = equalize_top(original_data_cpts)

    # Now extend depths to match the lowest_min_depth (equalized bottom)
    equalized_depth_cpts = equalize_depth(equalized_top_cpts, lowest_min_depth)

    # Print the results
    print(f"The lowest maximum depth is: {lowest_max_depth}")
    print(f"The lowest minimum depth is: {lowest_min_depth}")

    # Compress data to 32 points
    compressed_cpts = compress_to_32px(equalized_depth_cpts, method='mean')


    # Plot the original, equalized, and compressed data
    plot_equalized_depth_cpts(
        original_data_cpts,
        equalized_depth_cpts,
        compressed_cpts,
        num_to_plot=10,
        lowest_min_depth=lowest_min_depth,
        lowest_max_depth=lowest_max_depth,
    )

    # Plot the results of compression
    plot_compression_results(equalized_depth_cpts, compressed_cpts, num_to_plot=10)

    # Save the compressed data to a CSV file
    output_dir = r"D:\schemaGAN\real_case\groningen"
    save_cpt_to_csv(compressed_cpts, output_dir)
