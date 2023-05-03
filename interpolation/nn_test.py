# Importing necessary libraries
from scipy.interpolate import NearestNDInterpolator
import numpy as np
import matplotlib.pyplot as plt

# Creating a random number generator object
rng = np.random.default_rng()

# Generating two arrays of 10 random numbers each between -0.5 and 0.5
x = rng.random(10) - 0.5
y = rng.random(10) - 0.5

# Calculating the hypotenuse of (x, y) points
z = np.hypot(x, y)
print(z)

# Generating 1D arrays of X and Y values
X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))

# Creating a 2D grid using X and Y values
X, Y = np.meshgrid(X, Y)

# Creating a nearest-neighbors interpolation object with (x, y) as inputs and z as outputs
interp = NearestNDInterpolator(list(zip(x, y)), z)

# Interpolating Z values from the created grid using the nearest-neighbors algorithm
Z = interp(X, Y)

# Creating a 2D plot of the interpolated values as a color mesh
plt.pcolormesh(X, Y, Z, shading='auto')

# Plotting input (x, y) values as black circles
plt.plot(x, y, "ok", label="input point")

# Creating a legend for the plot
plt.legend()

# Creating a colorbar for the plot
plt.colorbar()

# Setting the aspect ratio to equal for a more accurate representation of the plot
plt.axis("equal")

# Displaying the plot
plt.show()
