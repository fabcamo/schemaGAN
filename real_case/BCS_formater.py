import csv

# Input and output file paths
input_file = r"D:\schemaGAN\data\BCS\missing\cptlike_81.csv"  # Replace with your actual CSV file path
output_file = r"D:\schemaGAN\data\BCS\bcs\cptlike_81_BCS.txt"  # Replace with your desired output file name

# Step 1: Read the CSV file and invert the z axis
with open(input_file, "r") as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)  # Read all rows into a list

    # Calculate the maximum z value for inversion
    max_z = max(float(row['z']) for row in data)

    # Invert the z-axis and store the modified data
    for row in data:
        z_original = float(row['z'])
        z_inverted = max_z - z_original  # Invert the z value
        row['z'] = z_inverted  # Update the z value in the row

# Step 2: Write the inverted z values and change format to scientific notation
with open(output_file, "w") as txtfile:
    for row in data:
        IC = float(row['IC'])
        if IC != 0:  # Exclude rows where IC is 0
            x1 = float(row['x'])  # Scale x by 10
            z_inverted = row['z']  # Scale inverted z by 10
            IC = IC  # Scale IC by 10
            txtfile.write(f"{x1:.18e} {z_inverted:.18e} {IC:.18e}\n")  # Use scientific notation

print(f"File converted, z-axis inverted, and saved as {output_file}.")




#
#
# import random
#
# # File path for the generated output
# output_file = r"D:\GeoSchemaGen\tests\BCS\test.txt"  # Replace with your desired file name
#
# # Open the file for writing
# with open(output_file, "w") as txtfile:
#     # Loop through x and y ranges, scaling to match the example
#     for x in range(1, 11):  # x from 1 to 10
#         for y in range(1, 11):  # y from 1 to 10
#             scaled_x = x * 10  # Scale x (e.g., 1 -> 10, 2 -> 20)
#             scaled_y = y * 10  # Scale y
#             value = random.uniform(1, 100)  # Random value between 1 and 100
#             txtfile.write(f"{scaled_x:.18e} {scaled_y:.18e} {value:.18e}\n")  # Use 18 decimal places in scientific notation
#
# print(f"Random coordinate file created: {output_file}")
#

