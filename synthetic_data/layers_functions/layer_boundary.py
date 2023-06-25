import random
import numpy as np
from random import betavariate
import matplotlib.pyplot as plt


# Generate a sine or cosine line as a layer boundary
def layer_boundary(x_coord, z_max):
    x_max = len(x_coord)
    amplitude = pert(2, 5, z_max)
    period = pert(x_max, 1000, 10000)
    phase_shift = np.random.uniform(low=0, high=x_max)
    vertical_shift = np.random.uniform(low=0, high=z_max)
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
from matplotlib import rcParams

# Set the font family to "Arial"
rcParams['font.family'] = 'Arial'

arr1 = [pert(2, 5, 32) for _ in range(10_000)]
arr2 = [pert(512, 1000, 10000) for _ in range(10_000)]
arr3 = np.random.uniform(0, 512, 10_000)
arr4 = np.random.uniform(0, 32, 10_000)

fig, axs = plt.subplots(2, 2, figsize=(9, 5))

axs[0, 0].hist(arr1, bins=50, alpha=0.5, color='silver', edgecolor='dimgray', density=True)
axs[0, 0].set_xlabel('(a)  Amplitude', fontsize=10)
axs[0, 0].set_ylabel('Relative Frequency', fontsize=10)
axs[0, 0].tick_params(axis='both', labelsize=8)

axs[0, 1].hist(arr2, bins=50, alpha=0.5, color='silver', edgecolor='dimgray', density=True)
axs[0, 1].set_xlabel('(b)  Period', fontsize=10)
axs[0, 1].set_ylabel('Relative Frequency', fontsize=10)
axs[0, 1].tick_params(axis='both', labelsize=8)

axs[1, 0].hist(arr3, bins=50, alpha=0.5, color='silver', edgecolor='dimgray', density=True)
axs[1, 0].set_xlabel('(c)  Phase Shift', fontsize=10)
axs[1, 0].set_ylabel('Relative Frequency', fontsize=10)
axs[1, 0].tick_params(axis='both', labelsize=8)

axs[1, 1].hist(arr4, bins=50, alpha=0.5, color='silver', edgecolor='dimgray', density=True)
axs[1, 1].set_xlabel('(d)  Vertical Shift', fontsize=10)
axs[1, 1].set_ylabel('Relative Frequency', fontsize=10)
axs[1, 1].tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.show()