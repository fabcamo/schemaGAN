import gstools as gs
import numpy as np


# Function to generate the random fields using the Gaussian model
def random_field_generator(std_value, mean, aniso_x, aniso_z, ndim, seed):
    len_scale = np.array([aniso_x, aniso_z])
    var = std_value**2

    model = gs.Gaussian(dim=ndim, var=var, len_scale=len_scale)
    srf = gs.SRF(model, mean=mean, seed=seed)

    return srf