import pickle
import numpy
with open('alti_results.pkl', 'rb') as f:
    alti_dict = pickle.load(f)

import matplotlib.pylab as plt

lists = sorted(alti_dict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples
y_src, y_tgt = zip(*y)


plt.plot(x, y_src, linestyle='-', marker='.', label='source')
plt.plot(x, y_tgt, linestyle='-', marker='.', label='target prefix')
plt.legend()
plt.title('ALTI+ mean source and target prefix contributions over the course of training', loc='center', wrap=True)
plt.grid(True)
plt.xlabel("# steps")
plt.ylabel("contribution")
plt.yticks(numpy.arange(0, 1., 0.1))
plt.savefig('alti_evolution.png')

plt.xscale('log')
plt.title('ALTI+ mean source and target prefix contributions over the course of training (lin-log)', loc='center', wrap=True)
ticks = 10**numpy.arange(2,6)
plt.xticks(ticks, ticks)
plt.savefig('alti_evolution_log.png')

