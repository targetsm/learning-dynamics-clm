import pickle

with open('alti_results_cpu.pkl', 'rb') as f:
    alti_dict = pickle.load(f)

import matplotlib.pylab as plt

lists = sorted(alti_dict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.savefig('alti_evolution.png')
plt.xscale('log')
plt.plot(x, y)
plt.savefig('alti_evolution_log.png')
