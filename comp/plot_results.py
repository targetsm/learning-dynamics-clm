import matplotlib.pylab as plt
import pickle
import numpy

with open('../kl/data/kl_dict.pkl', 'rb') as f:
    kl_dict_lm = pickle.load(f)
    list_lm = sorted([(k, v[0], v[1]) for k,v in kl_dict_lm.items() if int(k) >= 10000])
with open('../kl/data/kl_dict_unigram.pkl', 'rb') as f:
    kl_dict_unigram = pickle.load(f)
    list_unigram = sorted([(k, v[0], v[1]) for k,v in kl_dict_unigram.items() if int(k) >= 10000])
with open('../kl/data/kl_dict_bigram.pkl', 'rb') as f:
    kl_dict_bigram = pickle.load(f)
    list_bigram = sorted([(k, v[0], v[1]) for k,v in kl_dict_bigram.items() if int(k) >= 10000])

fig, axs = plt.subplots(6, 1, sharex=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.08)

ax1 = axs[0]
ax1.errorbar(*zip(*list_lm), capsize=4, linestyle='-', marker='.', label='lm')
#ax1.errorbar(*zip(*list_unigram), capsize=4, linestyle='-', marker='.', label='unigram')
#ax1.errorbar(*zip(*list_bigram), capsize=4, linestyle='-', marker='.', label='bigram')
ax1.legend()
ax1.grid(True)
ax1.set_ylabel("kl-divergence")

with open('../alti/transformer-contributions-nmt-v2/alti_results.pkl', 'rb') as f:
    alti_dict = pickle.load(f)
lists = sorted([(x, y) for x, y in alti_dict.items()  if int(x) >= 10000 and int(x) <= 100000]) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
y_src, y_tgt = zip(*y)

ax2 = axs[1]
ax2.plot(x, y_src, linestyle='-', marker='.', label='source contribution', color='red')
ax2.set_ylabel("contribution")
#ax2.set_yticks(numpy.arange(0.4, 1.1, 0.1))
ax2.grid()
ax2.legend()

valid_list = [x.split('\t') for x in open('./val/valid_list.txt', 'r').readlines()]
valid_list = [(int(y), float(x)) for [x,y] in valid_list if int(y) >= 10000]
train_list = [x.split('\t') for x in open('./val/train_list.txt', 'r').readlines()]
train_list = [(int(y), float(x)) for [x,y] in train_list if int(y) >= 10000]

ax3 = axs[2]
ax3.plot(*zip(*train_list), linestyle='-', marker='.', label='train loss')
ax3.plot(*zip(*valid_list), linestyle='-', marker='.', label='val loss')
ax3.set_ylabel("loss")
ax3.grid()
ax3.legend()

ax4 = axs[3]
with open('bleu/bleu.pkl', 'rb') as f:
    bleu_list = pickle.load(f)
lists = sorted([(x,y) for x, y in bleu_list.items() if int(x) >= 10000]) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
ax4.plot(x, y, linestyle='-', marker='.', label='test BLEU')
ax4.set_ylabel("BLEU")
ax4.grid()
ax4.legend()

ax5 = axs[4]
with open('comet/comet.pkl', 'rb') as f:
    comet_dict = pickle.load(f)
comet_dict = {int(ckpt):x.system_score for ckpt, x in comet_dict.items()  if int(ckpt) >= 10000}
lists = sorted(comet_dict.items()) # sorted by key, return a list of tuples
print(lists)
x, y = zip(*lists) # unpack a list of pairs into two tuples
ax5.plot(x, y, linestyle='-', marker='.', label='test COMET')
ax5.set_ylabel("COMET")
ax5.grid()
ax5.legend()

ax6 = axs[5]
with open('hallucinations/labse/labse.pkl', 'rb') as f:
    labse_dict = pickle.load(f)
labse_lowest = {int(ckpt):x[2] for ckpt, x in labse_dict.items() if int(ckpt) >= 10000}
labse_dict = {int(ckpt):x[0] for ckpt, x in labse_dict.items() if int(ckpt) >= 10000}
lists = sorted(labse_dict.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
ax6.plot(x, y, linestyle='-', marker='.', label='laBSE cos_sim')
#lists = sorted(labse_lowest.items())   
#x, y = zip(*lists) # unpack a list of pairs into two tuples
#ax6.plot(x, y, linestyle='-', marker='.', label='laBSE low 100')
ax6.set_ylabel("Similarity")
ax6.grid()
ax6.legend()

fig.suptitle('ALTI+ mean source contributions and KL divergence of TM and LM', wrap=True)

fig.set_size_inches(7,10)
plt.savefig('kl_plus_alti.pdf')

plt.xscale('log')
fig.suptitle('ALTI+ mean source contributions and KL divergence of TM and LM (lin-log)', wrap=True)
#ticks = [100, 500, 1000, 5000, 10000, 50000, 100000]
ticks = [10000, 50000, 60000, 100000]
plt.xticks(ticks, ticks)
plt.savefig('kl_plus_alti_log.pdf')

