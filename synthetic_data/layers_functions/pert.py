from random import betavariate
import matplotlib.pyplot as plt

def pert(a, b, c, *, lamb=4):
    r = c - a
    alpha = 1 + lamb * (b - a) / r
    beta = 1 + lamb * (c - b) / r
    return a + betavariate(alpha, beta) * r

arr = [pert(5, 50, 400) for _ in range(10_000)]

plt.hist(arr, bins=50)
plt.title('Example Histogram of PERT Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
