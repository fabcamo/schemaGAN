import os
import shutil
import time
import numpy as np


def split_data(data_path, train_folder, validation_folder, test_folder,
               vali_ratio=0.1666666, test_ratio=0.1666666, shuffle=True):
    """
    Split data into train, validation, and test sets.

    Args:
        data_path (str): Path to the source data directory.
        train_folder (str): Path to the directory where train data will be saved.
        validation_folder (str): Path to the directory where validation data will be saved.
        test_folder (str): Path to the directory where test data will be saved.
        vali_ratio (float, optional): Ratio of data for training (default is 0.1666666).
        test_ratio (float, optional): Ratio of data for testing (default is 0.1666666).
        shuffle (bool, optional): Whether to shuffle the data indices (default is True).
    """

    # Create directories if they don't exist
    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)
    if not os.path.isdir(validation_folder):
        os.makedirs(validation_folder)
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)

    # Get list of CSV files in data_path
    files = os.listdir(data_path)
    files = [f for f in files if f.endswith(".csv")]

    nb_files = len(files)
    nb_train = int(nb_files * vali_ratio)
    nb_test = int(nb_files * test_ratio)
    nb_validation = nb_files - (nb_train + nb_test)

    indexes = np.arange(nb_files)
    if shuffle:
        np.random.shuffle(indexes)

    # Select indexes for train, validation, and test sets
    indexes_train = indexes[:nb_train]
    indexes_validation = indexes[nb_train:nb_train + nb_validation]
    indexes_test = indexes[nb_train + nb_validation:]

    # Copy files to respective folders
    for i in indexes_train:
        shutil.copy(os.path.join(data_path, files[i]), os.path.join(train_folder, files[i]))

    for i in indexes_validation:
        shutil.copy(os.path.join(data_path, files[i]), os.path.join(validation_folder, files[i]))

    for i in indexes_test:
        shutil.copy(os.path.join(data_path, files[i]), os.path.join(test_folder, files[i]))

    # Delete files from the original folder
    for file_name in files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_path, file_name)
            os.remove(file_path)






def save_summary(output_folder, time_start, time_end, seed, no_realizations):
    execution_time = abs(time_start - time_end)
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    time_str = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

    time_current = time.strftime("%d/%m/%Y %H:%M:%S")
    file_path = os.path.join(output_folder, 'random_seed.txt')
    with open(file_path, 'w') as f:
        f.write("Executed on: {}\n".format(time_current))
        f.write("Execution time: {}\n\n".format(time_str))
        f.write("Seed: {}\n\n".format(seed))
        f.write("No. of realizations: {}\n\n".format(no_realizations))