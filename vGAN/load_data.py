import os
import pandas as pd

import tensorflow as tf


def combine_csv_files(directory):
    # Create an empty list to store the dataframes
    dfs = []

    # Loop through each CSV file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Read the CSV file into a dataframe and append it to the list
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            dfs.append(df)

    # Concatenate all dataframes in the list into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)

    # Return the combined dataframe
    return combined_df


# Set the directory containing the CSV files
directory = "D:\inpt\synthetic_data\cs2d_test"
# Call the function to combine the CSV files into a single dataframe
combined_df = combine_csv_files(directory)
# Print the combined dataframe
print(combined_df.shape)
