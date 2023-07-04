from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd


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



def process_cpts(cpts, file):
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
    data = []

    for i, cpt in enumerate(cpts):
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

        # convert arrays to string representation
        IC_str = ','.join(map(str, cpt_gef.IC))
        depth_str = ','.join(map(str, cpt_gef.depth_to_reference))
        coordinates_str = ','.join(map(str, cpt_gef.coordinates))

        data.append({"Name": f"CPT-{file+str(i)}",
                     "IC": IC_str,
                     "depth": depth_str,
                     "coordinates": coordinates_str})

    return pd.DataFrame(data)


def find_lowest_max_depth(df):
    """
    Find the maximum depth value for each CPT in a DataFrame,
    then find the lowest of these max values.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing CPT data

    Returns
    -------
    lowest_max_depth: dict
        Dictionary containing the name of the CPT and its lowest max depth
    """
    max_depths = {}

    for index, row in df.iterrows():
        # Convert string representation of list to actual list of floats
        depth_list = list(map(float, row['depth'].split(',')))

        # Find the maximum depth and store it in the dictionary
        max_depths[row['Name']] = max(depth_list)

    # Find the CPT name with the lowest max depth
    min_max_depth_name = min(max_depths, key=max_depths.get)

    lowest_max_depth = {min_max_depth_name: max_depths[min_max_depth_name]}

    return lowest_max_depth



def restructure_df(df):
    """
    Restructure a DataFrame of CPT data so that each row corresponds to one depth and its associated IC value.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing CPT data

    Returns
    -------
    new_df: pandas.DataFrame
        Restructured DataFrame
    """
    df_list = []

    for index, row in df.iterrows():
        # Convert string representations of lists to actual lists of floats
        depth_list = list(map(float, row['depth'].split(',')))
        IC_list = list(map(float, row['IC'].split(',')))

        # Create a DataFrame for this CPT
        cpt_df = pd.DataFrame({
            'Name': row['Name'],
            'depth': depth_list,
            'IC': IC_list
        })

        df_list.append(cpt_df)

    # Concatenate all DataFrames into a single DataFrame
    new_df = pd.concat(df_list, ignore_index=True)

    return new_df



if __name__ == "__main__":
    cpts_A = read_files(r"2422-220749_Sonderingen_GEF")
    data_cpts1 = process_cpts(cpts_A, file="A")
    cpts_B = read_files(r"cpts_BRO_in_embankment")
    data_cpts2 = process_cpts(cpts_B, file="B")

    # Concatenate the two dataframes
    all_cpts = pd.concat([data_cpts1, data_cpts2], ignore_index=True)

    # save combined dataframe to csv
    all_cpts.to_csv('CPT_data.csv', index=False)

    # Restructure the DataFrame
    only_IC = restructure_df(all_cpts)

    # save combined dataframe to csv
    only_IC.to_csv('ICdepth.csv', index=False)

    # Find CPT with the lowest max depth
    lowest_max_depth = find_lowest_max_depth(all_cpts)
    print(lowest_max_depth)


