import matplotlib.pylab as plt
import pickle
import numpy
import sys



fig, axs = plt.subplots(4, 2, sharex=False, constrained_layout=True)
fig.subplots_adjust(hspace=0.08)
dir_list = ['iwslt14deen/iwslt/', 'iwslt14deen/wmt/', 'wmt22deen_subset/iwslt/', 'wmt22frde_subset/iwslt/']
for i in range(len(dir_list)):
    dr = dir_list[i]
    with open(f'data/tl/{dr}/test_kl_dict.pkl', 'rb') as f:
        kl_dict_lm = pickle.load(f)
        list_lm = sorted([(k, v[0], v[1]) for k,v in kl_dict_lm.items()])
    with open(f'data/tl/{dr}/test_kl_dict_unigram.pkl', 'rb') as f:
        kl_dict_unigram = pickle.load(f)
        list_unigram = sorted([(k, v[0], v[1]) for k,v in kl_dict_unigram.items()])
    with open(f'data/tl/{dr}/test_kl_dict_bigram.pkl', 'rb') as f:
        kl_dict_bigram = pickle.load(f)
        list_bigram = sorted([(k, v[0], v[1]) for k,v in kl_dict_bigram.items()])

    name = dr.replace('/', '-').replace('_', '-').replace('wmt-big-', 'large').replace('wmt-', 'medium').replace('iwslt-', 'small')
    print(name)
    ax1 = axs[i][0]
    ax1.errorbar(*zip(*list_lm), capsize=4, linestyle='-', marker='.', label='lm')
    ax1.errorbar(*zip(*list_unigram), capsize=4, linestyle='-', marker='.', label='unigram')
    ax1.errorbar(*zip(*list_bigram), capsize=4, linestyle='-', marker='.', label='bigram')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(name)
    ax1.set_xlabel("# steps")
    ax1.set_ylabel("KL divergence")
    
    ax2 = axs[i][1]
    ax2.errorbar(*zip(*list_lm), capsize=4, linestyle='-', marker='.', label='lm')
    ax2.errorbar(*zip(*list_unigram), capsize=4, linestyle='-', marker='.', label='unigram')
    ax2.errorbar(*zip(*list_bigram), capsize=4, linestyle='-', marker='.', label='bigram')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel("# steps")
    ax2.set_ylabel("KL divergence")
    ax2.set_xscale('log')
    ax2.set_title(name + ' log')
    ticks = 10**numpy.arange(2,6)
    ax2.set_xticks(ticks, ticks)

#handles, labels = axs[3][1].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center')
fig.set_size_inches(8,12)
plt.savefig(f'plot/kl.pdf')


#plt.savefig(f'plot/{dr}/kl_log.pdf')
    


#with open('old_results/new_results/kl_dict.pkl', 'rb') as f:
#    kl_dict_lm_old = pickle.load(f)
#    list_lm_old = sorted([(k, v) for k,v in kl_dict_lm_old.items()])
#with open('old_results/new_results/kl_dict_unigram.pkl', 'rb') as f:
#    kl_dict_unigram_old = pickle.load(f)
#    list_unigram_old = sorted([(k, v) for k,v in kl_dict_unigram_old.items()])
#with open('old_results/new_results/kl_dict_bigram.pkl', 'rb') as f:
#    kl_dict_bigram_old = pickle.load(f)
#    list_bigram_old = sorted([(k, v) for k,v in kl_dict_bigram_old.items()])
#ax1.plot(*zip(*list_lm_old), linestyle='-', marker='.', label='lm_old')
#ax1.plot(*zip(*list_unigram_old), linestyle='-', marker='.', label='unigram_old')
#ax1.plot(*zip(*list_bigram_old), linestyle='-', marker='.', label='bigram_old')
#ax1.legend()
#plt.savefig('plot/kl_comp_log.pdf')

#with open('../alti/transformer-contributions-nmt-v2/alti_results.pkl', 'rb') as f:
#    alti_dict = pickle.load(f)
#lists = sorted(alti_dict.items()) # sorted by key, return a list of tuples
#x, y = zip(*lists) # unpack a list of pairs into two tuples
#y_src, y_tgt = zip(*y)
#
#plt.clf()
#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#plt.xscale('linear')
#ax1.plot(*zip(*list_lm), linestyle='-', marker='.', label='lm')
#ax2.plot(x, y_src, linestyle='-', marker='.', label='source contribution', color='red')
#ax2.set_ylabel("contribution")
#ax1.set_ylabel("KL divergence")
#ax2.set_yticks(numpy.arange(0.3, 1.1, 0.1))
#ax1.set_yticks(numpy.arange(0, 4.0, 0.5))
#ax1.grid(True, axis='both')
##ax2.grid(None)
#plt.title('ALTI+ mean source contributions and KL divergence of TM and LM', loc='center', wrap=True)
#ax1.legend()
#ax2.legend()
#plt.savefig('plot/kl_plus_alti.pdf')
#
#plt.xscale('log')
#plt.title('ALTI+ mean source contributions and KL divergence of TM and LM (lin-log)', loc='center', wrap=True)
#ticks = 10**numpy.arange(2,6)
#plt.xticks(ticks, ticks)
#plt.savefig('plot/kl_plus_alti_log.pdf')
#
