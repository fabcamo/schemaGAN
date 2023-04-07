import gstools as gs
import numpy as np


# This is not going to change for our project
ndim = 2
########################################################################################################################


def get_random_properties():
    std_value = 0.3
    mean = 2.1
    aniso_x = 40  # anisotropy in X
    aniso_z = 20  # anisotropy in Z
    angles = 0  # angle of rotation

    propeties = [std_value, mean, aniso_x, aniso_z, angles]


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