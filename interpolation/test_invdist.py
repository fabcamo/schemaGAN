import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Generate 4 random points in a 2D space
np.random.seed(42)
points = np.random.rand(4, 2) * 20

# Assign random values to the 4 points
values = np.random.randint(0, 10, size=4)

# Generate a 20x20 grid of points
xx, yy = np.meshgrid(np.arange(0, 20), np.arange(0, 20))
grid_points = np.column_stack((xx.ravel(), yy.ravel()))

# Build a KDTree from the 4 points
tree = KDTree(points)

# Calculate the inverse distances from each grid point to the 4 points
dists, indices = tree.query(grid_points, k=4)
weights = 1 / dists
weights /= weights.sum(axis=1)[:, np.newaxis]

# Interpolate the values for all the other points in the grid
interp_values = (values[indices] * weights).sum(axis=1)
interp_values = interp_values.reshape(xx.shape)

# Plot the original points and the interpolated surface
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

ax1.scatter(points[:, 0], points[:, 1], c=values, cmap='cool')
ax1.set_title('Input Points')
ax1.set_xlim([0, 20])
ax1.set_ylim([0, 20])
ax1.set_aspect('equal')


im = ax2.imshow(interp_values, origin='lower', cmap='cool', extent=[0, 20, 0, 20])
ax2.set_title('Interpolated Surface')
ax2.set_xlim([0, 20])
ax2.set_ylim([0, 20])
ax2.set_aspect('equal')
#plt.colorbar(im)
# Plot the input pixels on top of the interpolation results as black dots
ax2.scatter(points[:, 0], points[:, 1], c='black', s=30)



plt.show()
