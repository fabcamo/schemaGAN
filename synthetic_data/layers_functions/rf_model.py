import gstools as gs
import numpy as np

# Function to generate the random fields using the Gaussian model
def random_field_generator2(dim,var,len_scale,anis,angles,mean,seed):
    model = gs.Gaussian(
        dim=dim,  # dimension of the model
        var=var,  # variance of the model
        len_scale=len_scale,  # main length scale of the model
        anis=anis,  # transversal anisotropy
        angles=angles  # angle of rotation
    )
    srf = gs.SRF(model, mean=mean, seed=seed)
    return srf

# Function to generate the random fields using the Gaussian model
def random_field_generator(std_value, mean, aniso_x, aniso_z, ndim, seed):
    len_scale = np.array([aniso_x, aniso_z])
    var = std_value**2

    model = gs.Gaussian(dim=ndim, var=var, len_scale=len_scale)
    srf = gs.SRF(model, mean=mean, seed=seed)

    return srf