import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define custom path for CSV file
path = "cs_0.csv"

# Read CSV data using Pandas
df = pd.read_csv(path)
df_pivot = df.pivot(index='z', columns='x', values='IC')
print('the shape is:', df_pivot.shape)

# create a 2D image plot with x and z as axes and IC as pixel values
plt.imshow(df_pivot)
plt.axis('off')
plt.savefig('output.png', dpi=100, bbox_inches='tight', pad_inches=0)

# Create a 2D numpy array of random values with dimensions 256 by 64
data = np.random.rand(256, 64)

# Create a pandas dataframe from the numpy array
df = pd.DataFrame(data)
print('the df size is:', df)

# Print the first few rows of the dataframe to verify it was created successfully
print(df.head())