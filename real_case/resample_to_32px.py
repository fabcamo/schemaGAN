import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df_original = pd.read_csv(r'cptdata/all_cpt_with_zeros_at_bottom.csv')
# Create a copy of your dataframe for transformation
df = df_original.copy()

# Define the new min and max for the y (depth)
min_y_new = 0
max_y_new = 31

# Apply the linear transformation to the depth column
df['depth'] = df['depth'] * (max_y_new / df['depth'].max())

# Round the depth values to the nearest integer
df['depth'] = np.round(df['depth'])

# Group by the new depth and calculate the desired aggregations
aggregation_methods = ["mean", "geomean", "max"]

for method in aggregation_methods:
    if method == "mean":
        df_new = df.groupby('depth').mean().reset_index()
    elif method == "geomean":
        df_new = df.groupby('depth').apply(lambda x: np.exp(np.mean(np.log(x.iloc[:, 1:]), axis=0))).reset_index()
    elif method == "max":
        df_new = df.groupby('depth').max().reset_index()
    else:
        print("ERROR: That method does not exist.")
        continue

    # Get a list of your cpts
    cpts = df.columns[1:]  # we start from 1 to avoid 'depth'

    # Set the default font size
    plt.rcParams.update({'font.size': 6})

    # Create a subplot for each of the first 20 cpts
    fig, axs = plt.subplots(2, 20, figsize=(16, 6))

    # Create a subplot for each of the first 20 cpts for the original dataframe
    for i, cpt in enumerate(cpts[:20]):  # limit to first 20 cpts
        axs[0, i].plot(df_original[cpt], df_original['depth'], color='grey', alpha=0.8)
        axs[0, i].invert_yaxis()  # This line inverts the y-axis
        axs[0, i].set_title(f'{cpt}')
        axs[0, i].set_xlabel('IC')

        # Only add y-label for first subplot
        if i == 0:
            axs[0, i].set_ylabel('Z')
        else:
            axs[0, i].set_yticks([])  # This line removes the y-ticks

    # Create a combined subplot for each of the first 20 cpts
    for i, cpt in enumerate(cpts[:20]):  # limit to first 20 cpts
        # Plot original data
        axs[1, i].plot(df_original[cpt], df_original['depth'], color='grey', alpha=0.8)

        # Create a second y-axis
        axs_twin = axs[1, i].twinx()

        # Plot new data on second y-axis
        axs_twin.plot(df_new[cpt], df_new['depth'], color='black', alpha=0.8)
        axs_twin.invert_yaxis()  # This line inverts the y-axis for the transformed data

        axs[1, i].invert_yaxis()  # This line inverts the y-axis for the original data
        axs[1, i].set_title(f'{cpt} ({method})')
        axs[1, i].set_xlabel('IC')

        # Only add y-label for first subplot
        if i == 0:
            axs[1, i].set_ylabel('Z (original)')
            axs_twin.set_ylabel('Z (transformed)')
        else:
            axs[1, i].set_yticks([])  # This line removes the y-ticks
            axs_twin.set_yticks([])

    # Automatically adjust subplot parameters to give specified padding
    plt.tight_layout()

    # Save the plot as a PDF file with the same name as the CSV file
    output_filename = r'cptdata/32px_allCPT_{}.pdf'.format(method)
    plt.savefig(output_filename)

    # Save the new DataFrame to a CSV file
    output_filename_csv = r'cptdata/32px_allCPT_{}.csv'.format(method)
    df_new.to_csv(output_filename_csv, index=False)

    # Show the plot (optional)
    plt.show()
