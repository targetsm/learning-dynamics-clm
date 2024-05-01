import matplotlib.pyplot as plt
import numpy as np
import sys

x = [float(y) for y in open(sys.argv[1], 'r').readlines()]
print(x)
plt.hist(x, density=False, bins=30)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.savefig(f'{sys.argv[1]}_dist.png')
