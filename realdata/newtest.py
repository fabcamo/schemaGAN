import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df_original = pd.read_csv('cptfix.csv')

# Create a copy of your dataframe for transformation
df = df_original.copy()

# Define the new min and max for the y (depth)
min_y_new = 0
max_y_new = 32

# Apply the linear transformation to the depth column
df['depth'] = df['depth'] * (max_y_new / df['depth'].max())

# Round the depth values to the nearest integer
df['depth'] = np.round(df['depth'])

# Group by the new depth and average the other columns
df_new = df.groupby('depth').mean().reset_index()

# Get a list of your cpts
cpts = df.columns[1:]  # we start from 1 to avoid 'depth'

# Set the default font size
plt.rcParams.update({'font.size': 6})

# Create a subplot for each cpt for original and new dataframes
fig, axs = plt.subplots(2, len(cpts), figsize=(30, 6))

# Create a separate subplot for each cpt for the original dataframe
for i, cpt in enumerate(cpts):
    axs[0, i].plot(df_original[cpt], df_original['depth'], color='rebeccapurple')
    axs[0, i].invert_yaxis()  # This line inverts the y-axis
    axs[0, i].set_title(f'{cpt}')
    axs[0, i].set_xlabel('IC')

    # Only add y-label for first subplot
    if i == 0:
        axs[0, i].set_ylabel('Z')
    else:
        axs[0, i].set_yticks([])  # This line removes the y-ticks

# Create a separate subplot for each cpt for the new dataframe
for i, cpt in enumerate(cpts):
    axs[1, i].plot(df_new[cpt], df_new['depth'], color='darkgreen')
    axs[1, i].invert_yaxis()  # This line inverts the y-axis
    axs[1, i].set_title(f'{cpt}')
    axs[1, i].set_xlabel('IC')

    # Only add y-label for first subplot
    if i == 0:
        axs[1, i].set_ylabel('Z')
    else:
        axs[1, i].set_yticks([])  # This line removes the y-ticks

# Automatically adjust subplot parameters to give specified padding
plt.tight_layout()

# Show the plot
plt.show()

# Save the new DataFrame to a csv file
df_new.to_csv('newdepths_cptfix.csv', index=False)