import pickle
import numpy
import sys

dr = sys.argv[1]
with open(f'plt/{dr}/results.pkl', 'rb') as f:
    results_dict = pickle.load(f)

import matplotlib.pylab as plt
fig, ax = plt.subplots(nrows=1, sharex=True)

lists = sorted(results_dict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

halu_mean = [numpy.mean(j) for j in y]
halu_se = [numpy.std(j)/numpy.sqrt(len(j)) for j in y]

ax.errorbar(x, halu_mean, halu_se, capsize=4, linestyle='-', marker='.', label='hallucination ratio')
#plt.legend()
ax.set_title('Ratio of hallucinated tokens over the course of training', loc='center', wrap=True)
ax.grid(True)
ax.set_xlabel("# steps")
ax.set_ylabel("ratio")
#plt.yticks(numpy.arange(0, 1., 0.1))
ax.set_yticks(numpy.arange(0, 1., 0.1))
fig.savefig(f'plt/{dr}/halu_evolution_total.pdf')

ax.set_xscale('log')
ax.set_title('Ratio of hallucinated tokens over the course of training (lin-log)', loc='center', wrap=True)
ticks = 10**numpy.arange(2,6)
ax.set_xticks(ticks, ticks)
fig.savefig(f'plt/{dr}/halu_evolution_total_log.pdf')

