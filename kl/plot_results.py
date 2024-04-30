import matplotlib.pylab as plt
import pickle
import numpy

with open('kl_dict.pkl', 'rb') as f:
    kl_dict_lm = pickle.load(f)
    list_lm = sorted(kl_dict_lm.items()) # sorted by key, return a list of tuples
with open('kl_dict_unigram.pkl', 'rb') as f:
    kl_dict_unigram = pickle.load(f)
    list_unigram = sorted(kl_dict_unigram.items())
with open('kl_dict_bigram.pkl', 'rb') as f:
    kl_dict_bigram = pickle.load(f)
    list_bigram = sorted(kl_dict_bigram.items())

fig, ax1 = plt.subplots()
ax1.plot(*zip(*list_lm), linestyle='-', marker='.', label='lm')
ax1.plot(*zip(*list_unigram), linestyle='-', marker='.', label='unigram')
ax1.plot(*zip(*list_bigram), linestyle='-', marker='.', label='bigram')
ax1.legend()
ax1.grid(True)
plt.title("Test set KL divergence of TM & LM after the same amount \n of training steps")
ax1.set_xlabel("# steps")
ax1.set_ylabel("kl-divergence")
plt.savefig('kl.png')
plt.xscale('log')
plt.title("Test set KL divergence of TM & LM after the same amount \n of training steps (lin-log)")
ticks = 10**numpy.arange(2,6)
plt.xticks(ticks, ticks)
plt.savefig('kl_log.png')

with open('../alti/transformer-contributions-nmt-v2/alti_results.pkl', 'rb') as f:
    alti_dict = pickle.load(f)
lists = sorted(alti_dict.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
y_src, y_tgt = zip(*y)

plt.clf()
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
plt.xscale('linear')
ax1.plot(*zip(*list_lm), linestyle='-', marker='.', label='lm')
ax2.plot(x, y_src, linestyle='-', marker='.', label='source contribution', color='red')
ax2.set_ylabel("contribution")
ax1.set_ylabel("KL divergence")
ax2.set_yticks(numpy.arange(0.3, 1.1, 0.1))
ax1.set_yticks(numpy.arange(0, 4.0, 0.5))
ax1.grid(True, axis='both')
#ax2.grid(None)
plt.title('ALTI+ mean source contributions and KL divergence of TM and LM', loc='center', wrap=True)
ax1.legend()
ax2.legend()
plt.savefig('kl_plus_alti.png')

plt.xscale('log')
plt.title('ALTI+ mean source contributions and KL divergence of TM and LM (lin-log)', loc='center', wrap=True)
ticks = 10**numpy.arange(2,6)
plt.xticks(ticks, ticks)
plt.savefig('kl_plus_alti_log.png')

