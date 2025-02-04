import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define input and output directories
input_folder = r"D:\schemaGAN\data\compare"
output_folder = r"D:\schemaGAN\tests\ims"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get list of all CSV files in the input directory
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

# Sort files to process them in order
csv_files.sort()

for csv_file in csv_files:
    file_path = os.path.join(input_folder, csv_file)
    output_path = os.path.join(output_folder, csv_file.replace(".csv", ".png"))

    # Load CSV file, skipping first row which contains column names
    df = pd.read_csv(file_path, skiprows=1, names=[" ", "x", "z", "IC"])

    # Pivot data to create a 2D array of size (32, 512)
    grid = df.pivot(index="z", columns="x", values="IC")

    # Sort axes to maintain proper orientation
    grid = grid.sort_index(ascending=True)

    # Plot the image
    plt.figure(figsize=(5, 5))  # Set equal aspect ratio
    plt.imshow(grid, cmap='viridis', aspect='equal', origin='lower')
    plt.axis('off')  # Remove axis, labels, and title

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

print("All images have been processed and saved.")