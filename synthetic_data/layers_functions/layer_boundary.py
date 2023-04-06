import random
import numpy as np
from layers_functions.pert import pert
def layer_boundary(x_coord):

    amplitude = pert(2,10,60)
    period = pert(400, 1000, 6000)
    phase_shift = np.random.uniform(low=0, high=500)
    vertical_shift = np.random.uniform(low=0, high=60)
    func = random.choice([np.sin, np.cos])
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y