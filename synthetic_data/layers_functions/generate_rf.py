import gstools as gs
import numpy as np


# This is not going to change for our project
ndim = 2


########################################################################################################################
# Define the Random Field parameters for each material
# > The mean and std defined per material according to Robertson
# > The other parameters defined at random
# Robertson, P.K., 1990. Soil classification using the cone penetration test. Canadian Geotechnical Journal, 27(1): 151-158.
# Robertson, P.K., 2009a. Interpretation of cone penetration tests – a unified approach. Canadian Geotechnical Journal, 46:1337-1355.
# Robertson, P. K., Cabal, K. (2022). Guide to Cone Penetration Testing (7th ed.). Gregg Drilling LLC.


# RF-IC values for clay
def soil_behaviour_clay():
    min_IC = 2.95
    max_IC = 3.6
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(10,40)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(4, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles


# RF-IC values for clayey silt to silty clay
def soil_behaviour_siltmix():
    min_IC = 2.6
    max_IC = 2.95
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(10,40)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(4, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles


# RF-IC values for silty sand to sandy silt
def soil_behaviour_sandmix():
    min_IC = 2.05
    max_IC = 2.6
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(10,40)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(4, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles


# RF-IC values for sand
def soil_behaviour_sand():
    min_IC = 1.31
    max_IC = 2.05
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(10,40)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(4, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles


# RF-IC values for organic soils
def soil_behaviour_organic():
    min_IC = 3.6
    max_IC = 4.2
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(10,40)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(4, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles



# Function to generate the random fields using the Gaussian model
def rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed):
    len_scale = np.array([aniso_x, aniso_z])
    var = std_value**2

    model = gs.Gaussian(dim=ndim, var=var, len_scale=len_scale, angles=angles)
    srf = gs.SRF(model, mean=mean, seed=seed)

    return srf


# generate the random field models for different materials
def generate_rf_group(seed):

    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_clay()
    srf_clay = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 1)
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_siltmix()
    srf_siltmix = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 2)
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_sandmix()
    srf_sandmix = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 3)
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_sand()
    srf_sand = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 4)
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_organic()
    srf_organic = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 5)
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_clay()
    srf_clay2 = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 6)
    std_value, mean, aniso_x, aniso_z, angles = soil_behaviour_sand()
    srf_sand2 = rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed + 7)

    # store the random field models inside layers
    layers = [srf_clay, srf_siltmix, srf_sandmix, srf_sand, srf_organic, srf_clay2, srf_sand2]

    return layers