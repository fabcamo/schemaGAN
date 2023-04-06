from random import betavariate
import matplotlib.pyplot as plt

def pert(low, peak, high, *, lamb=10):
    r = high - low
    alpha = 1 + lamb * (peak - low) / r
    beta = 1 + lamb * (high - peak) / r
    return low + betavariate(alpha, beta) * r

#low = 2
#peak = 10
#high = 90

#arr = [pert(low, peak, high) for _ in range(10_000)]

#n_pert = pert(low, peak, high)
#print('a random number is beta is>', n_pert)

#plt.hist(arr, bins=50)
#plt.title('Example Histogram of PERT Distribution')
#plt.xlabel('Values')
#plt.ylabel('Frequency')
#plt.show()
