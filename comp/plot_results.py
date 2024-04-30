import matplotlib.pylab as plt
import pickle
import numpy

with open('../kl/kl_dict.pkl', 'rb') as f:
    kl_dict_lm = pickle.load(f)
    list_lm = sorted(kl_dict_lm.items()) # sorted by key, return a list of tuples
with open('../kl/kl_dict_unigram.pkl', 'rb') as f:
    kl_dict_unigram = pickle.load(f)
    list_unigram = sorted(kl_dict_unigram.items())
with open('../kl/kl_dict_bigram.pkl', 'rb') as f:
    kl_dict_bigram = pickle.load(f)
    list_bigram = sorted(kl_dict_bigram.items())

fig, axs = plt.subplots(3, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.08)

ax1 = axs[0]
ax1.plot(*zip(*list_lm), linestyle='-', marker='.', label='lm')
ax1.plot(*zip(*list_unigram), linestyle='-', marker='.', label='unigram')
ax1.plot(*zip(*list_bigram), linestyle='-', marker='.', label='bigram')
ax1.legend()
ax1.grid(True)
ax1.set_ylabel("kl-divergence")

with open('../alti/transformer-contributions-nmt-v2/alti_results.pkl', 'rb') as f:
    alti_dict = pickle.load(f)
lists = sorted(alti_dict.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
y_src, y_tgt = zip(*y)

ax2 = axs[1]
ax2.plot(x, y_src, linestyle='-', marker='.', label='source contribution', color='red')
ax2.set_ylabel("contribution")
ax2.set_yticks(numpy.arange(0.4, 1.1, 0.1))
ax2.grid()
ax2.legend()


valid_list = [x.split('\t') for x in open('./val/valid_list.txt', 'r').readlines()]
valid_list = [(int(y), float(x)) for [x,y] in valid_list]
train_list = [x.split('\t') for x in open('./val/train_list.txt', 'r').readlines()]
train_list = [(int(y), float(x)) for [x,y] in train_list]

ax3 = axs[2]
ax3.plot(*zip(*train_list), linestyle='-', marker='.', label='train loss')
ax3.plot(*zip(*valid_list), linestyle='-', marker='.', label='val loss')
ax3.set_ylabel("loss")
ax3.grid()
ax3.legend()


fig.suptitle('ALTI+ mean source contributions and KL divergence of TM and LM', wrap=True)

fig.set_size_inches(7,7)
plt.savefig('kl_plus_alti.pdf')

plt.xscale('log')
fig.suptitle('ALTI+ mean source contributions and KL divergence of TM and LM (lin-log)', wrap=True)
ticks = [100, 500, 1000, 5000, 10000, 50000, 100000]
plt.xticks(ticks, ticks)
plt.savefig('kl_plus_alti_log.pdf')



