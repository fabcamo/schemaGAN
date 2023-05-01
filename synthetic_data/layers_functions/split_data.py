import os
import shutil
import numpy as np

# Split data into train and validation set
# From Bruno's randomlayers
def split_data(data_path, train_folder, validation_folder, train_size, shuffle=True):

    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)

    if not os.path.isdir(validation_folder):
        os.makedirs(validation_folder)

    files = os.listdir(data_path)
    files = [f for f in files if f.endswith(".csv")]

    nb_files = len(files)
    nb_train = int(nb_files * train_size)
    indexes_train = np.random.choice(range(nb_files), nb_train, replace=False)
    indexes_validation = np.array([i for i in range(nb_files) if i not in indexes_train])

    for i in indexes_train:
        shutil.copy(os.path.join(data_path, files[i]), os.path.join(train_folder, files[i]))

    for i in indexes_validation:
        shutil.copy(os.path.join(data_path, files[i]), os.path.join(validation_folder, files[i]))



'''
output_folder = 'C:\\inpt\\synthetic_data\\512x32_100k'
# Split the data into train and validation
split_data(output_folder, os.path.join(output_folder, "train"),
           os.path.join(output_folder, "./validation"), train_size=0.8)
'''