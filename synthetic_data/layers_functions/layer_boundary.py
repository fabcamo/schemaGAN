import random
import numpy as np
from random import betavariate
import matplotlib.pyplot as plt


# Generate a sine or cosine line as a layer boundary
def layer_boundary(x_coord):

    amplitude = pert(2,5,60)
    period = pert(400, 1000, 10000)
    phase_shift = np.random.uniform(low=0, high=500)
    vertical_shift = np.random.uniform(low=0, high=60)
    func = random.choice([np.sin, np.cos])
    y = amplitude * func(2 * np.pi * (x_coord - phase_shift) / period) + vertical_shift

    return y


# Function to generate a Beta-Pert distribution
def pert(low, peak, high, *, lamb=10):
    r = high - low
    alpha = 1 + lamb * (peak - low) / r
    beta = 1 + lamb * (high - peak) / r
    return low + betavariate(alpha, beta) * r


########################################################################################################################
# To plot the Pert graph
'''
low = 400
peak = 1000
high = 10000

arr = [pert(low, peak, high) for _ in range(10_000)]

plt.hist(arr, bins=50)
plt.title('Histogram of PERT Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
'''