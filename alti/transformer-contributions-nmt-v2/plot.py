import pickle
import numpy
import sys

dr = sys.argv[1]
with open(f'plt/{dr}/alti_results.pkl', 'rb') as f:
    alti_dict = pickle.load(f)

import matplotlib.pylab as plt

lists = sorted(alti_dict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples
#print(x)
src = [el['src'] for el in y]
trg = [el['trg'] for el in y]
#print(src)
#y_src, y_tgt = zip(*y)
src_mean = [numpy.mean(el) for el in src]
src_se = [numpy.std(el)/numpy.sqrt(len(src)) for el in src]
trg_mean = [numpy.mean(el) for el in trg]
trg_se = [numpy.std(el)/numpy.sqrt(len(trg)) for el in trg]

#plt.plot(x, y_src, linestyle='-', marker='.', label='source')
#plt.plot(x, y_tgt, linestyle='-', marker='.', label='target prefix')
plt.errorbar(x, src_mean, src_se, capsize=4, linestyle='-', marker='.', label='source')
plt.errorbar(x, trg_mean, trg_se, capsize=4, linestyle='-', marker='.', label='target prefix')

plt.legend()
plt.title('ALTI+ mean source and target prefix contributions over the course of training', loc='center', wrap=True)
plt.grid(True)
plt.xlabel("# steps")
plt.ylabel("contribution")
plt.yticks(numpy.arange(0, 1., 0.1))
plt.savefig(f'plt/{dr}/alti_evolution.pdf')

plt.xscale('log')
plt.title('ALTI+ mean source and target prefix contributions over the course of training (lin-log)', loc='center', wrap=True)
ticks = 10**numpy.arange(2,6)
plt.xticks(ticks, ticks)
plt.savefig(f'plt/{dr}/alti_evolution_log.pdf')

