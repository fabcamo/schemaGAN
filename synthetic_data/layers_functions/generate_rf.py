import gstools as gs
import numpy as np


# This is not going to change for our project
ndim = 2
########################################################################################################################
# Robertson, P.K., 1990. Soil classification using the cone penetration test. Canadian Geotechnical Journal, 27(1): 151-158.
# Robertson, P.K., 2009a. Interpretation of cone penetration tests – a unified approach. Canadian Geotechnical Journal, 46:1337-1355.
# Robertson, P. K., Cabal, K. (2022). Guide to Cone Penetration Testing (7th ed.). Gregg Drilling LLC.



# RF-IC values for clay
def soil_behaviour_clay():
    min_IC = 2.95
    max_IC = 3.6
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(20,80)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(2, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles

# RF-IC values for clayey silt to silty clay
def soil_behaviour_siltmix():
    min_IC = 2.6
    max_IC = 2.95
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(20,80)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(2, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles

# RF-IC values for silty sand to sandy silt
def soil_behaviour_sandmix():
    min_IC = 2.05
    max_IC = 2.6
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(20,80)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(2, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles

# RF-IC values for sand
def soil_behaviour_sand():
    min_IC = 1.31
    max_IC = 2.05
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(20,80)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(2, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles


# RF-IC values for organic soils
def soil_behaviour_organic():
    min_IC = 3.6
    max_IC = 4.2
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.randint(20,80)  # anisotropy in X
    aniso_z = aniso_x / np.random.randint(2, 10)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles



# Function to generate the random fields using the Gaussian model
def random_field_generator(std_value, mean, aniso_x, aniso_z, angles, seed):
    len_scale = np.array([aniso_x, aniso_z])
    var = std_value**2

    model = gs.Gaussian(dim=ndim, var=var, len_scale=len_scale, angles=angles)
    srf = gs.SRF(model, mean=mean, seed=seed)

    return srf


# generate the random field models for different materials
def generate_rf_group(aniso_x, aniso_z, angles, seed):

    srf_sand = random_field_generator(0.3, 1.5, aniso_x, aniso_z, angles, seed+1)
    srf_clay = random_field_generator(0.3, 2.1, aniso_x, aniso_z, angles, seed+2)
    srf_silt = random_field_generator(0.5, 3.2, aniso_x, aniso_z, angles, seed+3)
    # store the random field models inside layers
    layers = [srf_sand, srf_silt, srf_clay, srf_sand, srf_clay, srf_silt, srf_sand]

    return layers