import gstools as gs
import numpy as np

# This is not going to change for our project
ndim = 2

def soil_behaviour_clay():
    """
    Return RF-IC values for clay.

    Returns:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    min_IC = 2.95
    max_IC = 3.6
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.triangular(3, 47, 80)  # anisotropy in X
    aniso_z = np.random.uniform(0.2, 3)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles



def soil_behaviour_siltmix():
    """
    Return RF-IC values for clayey silt to silty clay.

    Returns:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    min_IC = 2.6
    max_IC = 2.95
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.triangular(3, 47, 80)  # anisotropy in X
    aniso_z = np.random.uniform(0.2, 3)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles



def soil_behaviour_sandmix():
    """
    Return RF-IC values for silty sand to sandy silt.

    Returns:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    min_IC = 2.05
    max_IC = 2.6
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.triangular(3, 47, 80)  # anisotropy in X
    aniso_z = np.random.uniform(0.2, 3)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles



def soil_behaviour_sand():
    """
    Return RF-IC values for sand.

    Returns:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    min_IC = 1.31
    max_IC = 2.05
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.triangular(3, 47, 80)  # anisotropy in X
    aniso_z = np.random.uniform(0.2, 3)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles



def soil_behaviour_organic():
    """
    Return RF-IC values for organic soils.

    Returns:
        Tuple: Tuple containing std_value, mean, aniso_x, aniso_z, and angles.
    """
    min_IC = 3.6
    max_IC = 4.2
    mean = (min_IC + max_IC) / 2
    std_value = (max_IC - min_IC) / 6
    aniso_x = np.random.triangular(3, 47, 80)  # anisotropy in X
    aniso_z = np.random.uniform(0.2, 3)  # anisotropy in Z
    angle_factor = np.random.triangular(20, 80, 100)
    angles = np.pi / angle_factor  # angle of rotation
    return std_value, mean, aniso_x, aniso_z, angles



def rf_generator(std_value, mean, aniso_x, aniso_z, angles, seed):
    """
    Generate random fields using the Gaussian model.

    Args:
        std_value (float): Standard deviation value.
        mean (float): Mean value.
        aniso_x (float): Anisotropy value in X direction.
        aniso_z (float): Anisotropy value in Z direction.
        angles (float): Angle of rotation.
        seed (int): Random seed.

    Returns:
        gs.SRF: Generated random field model.
    """
    len_scale = np.array([aniso_x, aniso_z])
    var = std_value**2

    model = gs.Gaussian(dim=ndim, var=var, len_scale=len_scale, angles=angles)
    srf = gs.SRF(model, mean=mean, seed=seed)

    return srf



def generate_rf_group(seed):
    """
    Generate random field models for different materials.

    Args:
        seed (int): Random seed.

    Returns:
        list: List of generated random field models.
    """
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
