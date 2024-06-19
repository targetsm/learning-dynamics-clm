import pickle
import numpy
with open('results_first_500.pkl', 'rb') as f:
    results_dict = pickle.load(f)

import matplotlib.pylab as plt

lists = sorted(results_dict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples


plt.plot(x, y, linestyle='-', marker='.')
#plt.legend()
plt.title('Ratio of hallucinated tokens over the course of training', loc='center', wrap=True)
plt.grid(True)
plt.xlabel("# steps")
plt.ylabel("Ratio")
#plt.yticks(numpy.arange(0, 1., 0.1))
plt.savefig('halu_evolution_total.pdf')

plt.xscale('log')
plt.title('Ratio of hallucinated tokens over the course of training (lin-log)', loc='center', wrap=True)
ticks = 10**numpy.arange(2,6)
plt.xticks(ticks, ticks)
plt.savefig('halu_evolution_total_log.pdf')

