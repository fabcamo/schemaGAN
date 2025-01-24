import os
import re
import pandas as pd


def extract_properties_from_filename(filename):
    # Regular expression to extract properties only up to POP
    pattern = r"(?P<Material>[\w]+)_(?P<Material_Type>[\w]+)_(?P<Cohesion>\d+KPa)_Height(?P<Height>\d+)Depth(?P<Depth>\d+)Beta(?P<Beta>\d+)Gamma(?P<Gamma>\d+)Alpha(?P<Alpha>[\d\.]+)POP(?P<POP>\d+)_"

    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    else:
        return None


def extract_properties_from_folder(folder_path):
    # List to store all extracted properties
    extracted_data = []

    # Traverse through the folder and process .p3d files
    for filename in os.listdir(folder_path):
        if filename.endswith(".p3d"):
            properties = extract_properties_from_filename(filename)
            if properties:
                extracted_data.append(properties)

    return extracted_data


def consolidate_data(extracted_data):
    # Dictionary to store the consolidated data
    consolidated = {
        'Material': [],
        'Material_Type': [],
        'Cohesion': set(),
        'Height': set(),
        'Depth': set(),
        'Beta': set(),
        'Gamma': set(),
        'Alpha': set(),
        'POP': set()
    }

    # Consolidate the data
    for entry in extracted_data:
        for key in consolidated:
            if isinstance(consolidated[key], set):
                consolidated[key].add(entry[key])
            else:
                consolidated[key] = entry[key]

    # Convert sets to sorted lists
    consolidated = {key: sorted(list(value)) if isinstance(value, set) else value for key, value in consolidated.items()}

    return consolidated


def process_folders_to_single_file(folder_paths, output_file):
    # List to store consolidated data for all folders
    all_consolidated_data = []

    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        extracted_data = extract_properties_from_folder(folder_path)

        if extracted_data:
            consolidated_data = consolidate_data(extracted_data)
            consolidated_data['Path'] = folder_path  # Add the folder path as a column
            all_consolidated_data.append(consolidated_data)
        else:
            print(f"No valid .p3d files found in the folder: {folder_path}")

    # Create a DataFrame from all the consolidated data
    df = pd.DataFrame(all_consolidated_data)

    # Save the DataFrame to a single CSV file
    df.to_csv(output_file, index=False)
    print(f"Consolidated data saved to {output_file}")


def main():
    # List of folder paths to process
    folder_paths = [
        r"N:\Projects\11207000\11207168\B. Measurements and calculations\Jaar 3\2c-1 3D-2D FEM embankment modeling\PLX SHANSEP\All PLX models\C2\1mSand",
        r"N:\Projects\11207000\11207168\B. Measurements and calculations\Jaar 3\2c-1 3D-2D FEM embankment modeling\PLX SHANSEP\All PLX models\C2\FullSand",
        r"N:\Projects\11207000\11207168\B. Measurements and calculations\Jaar 3\2c-1 3D-2D FEM embankment modeling\PLX SHANSEP\All PLX models\D4\D4 1m",
        r"N:\Projects\11207000\11207168\B. Measurements and calculations\Jaar 3\2c-1 3D-2D FEM embankment modeling\PLX SHANSEP\All PLX models\D4\D4 full",
        r"N:\Projects\11207000\11207168\B. Measurements and calculations\Jaar 3\2c-1 3D-2D FEM embankment modeling\PLX SHANSEP\All PLX models\VIRM4\1m",
        r"N:\Projects\11207000\11207168\B. Measurements and calculations\Jaar 3\2c-1 3D-2D FEM embankment modeling\PLX SHANSEP\All PLX models\VIRM4\full"
    ]

    # Define the single output file
    output_file = r"C:\PLX_Consolidated_Properties\all_consolidated_properties.csv"

    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process folders and save the data to the single CSV file
    process_folders_to_single_file(folder_paths, output_file)


if __name__ == "__main__":
    main()
