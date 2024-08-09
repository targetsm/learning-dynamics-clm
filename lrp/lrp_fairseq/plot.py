import pickle
import numpy
import sys
dr = sys.argv[1]
with open(f'{dr}/lrp_results.pkl', 'rb') as f:
    total_dict = pickle.load(f)

import matplotlib.pylab as plt
#print(total_dict['100000.pt'])
print(total_dict)
lrp_dict = {int(key.split('.')[0].split('_')[-1]): (d['inp'], d['out']) for key, d in total_dict.items()}
print(lrp_dict)
lists = sorted(lrp_dict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples
src = [el[0] for el in y]
trg = [el[1] for el in y]
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
plt.title('LRP mean source and target prefix contributions over the course of training', loc='center', wrap=True)
plt.grid(True)
plt.xlabel("# steps")
plt.ylabel("contribution")
#plt.yticks(numpy.arange(0, 1., 0.1))
plt.savefig(f'{dr}/lrp_evolution.pdf')

plt.xscale('log')
plt.title('LRP mean source and target prefix contributions over the course of training (lin-log)', loc='center', wrap=True)
ticks = 10**numpy.arange(2,6)
plt.xticks(ticks, ticks)
plt.savefig(f'{dr}/lrp_evolution_log.pdf')

