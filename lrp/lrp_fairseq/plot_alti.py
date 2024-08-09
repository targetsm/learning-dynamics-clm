import pickle
import numpy
import sys
import matplotlib.pyplot as plt


dir_list = ['iwslt14deen/iwslt/', 'iwslt14deen/wmt/']
fig, axs = plt.subplots(4, 2, sharex=False, constrained_layout=True)
fig.subplots_adjust(hspace=0.08)
for i in range(len(dir_list)):
    dr = dir_list[i]
    with open(f'/local/home/ggabriel/ma/alti/transformer-contributions-nmt-v2/plt/tl/{dr}/alti_results.pkl', 'rb') as f:
        alti_dict = pickle.load(f)
    lists = [(x, y) for x, y in sorted(alti_dict.items()) if x <= 100000] # sorted by key, return a list of tuples
    print(dr) 
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    #print(x)
    #print(y)
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
    name = dr.replace('/', '-').replace('_', '-').replace('wmt-big-', 'large').replace('wmt-', 'medium').replace('iwslt-', 'small')
    print(name)
    ax0 = axs[i*2][0]
    ax0.errorbar(x, src_mean, src_se, capsize=4, linestyle='-', marker='.', label='source')
    ax0.errorbar(x, trg_mean, trg_se, capsize=4, linestyle='-', marker='.', label='target prefix')
    ax0.legend()
    ax0.grid(True)
    ax0.set_ylabel("contribution")
    #ax0.set_xlabel("# steps")
    ax0.set_yticks(numpy.arange(0, 1., 0.1))
    ax0.set_title(name+ ' ALTI+')
    ax1 = axs[i*2][1]
    ax1.errorbar(x, src_mean, src_se, capsize=4, linestyle='-', marker='.', label='source')
    ax1.errorbar(x, trg_mean, trg_se, capsize=4, linestyle='-', marker='.', label='target prefix')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylabel("contribution")
    #ax1.set_xlabel("# steps")
    ax1.set_yticks(numpy.arange(0, 1., 0.1))
    ax1.set_title(name + ' ALTI+ (log)')
    ax1.set_xscale('log')
    ticks = 10**numpy.arange(2,6)
    ax1.set_xticks(ticks, ticks)


    with open(f'plt/{dr}/lrp_results.pkl', 'rb') as f:
        total_dict = pickle.load(f)

    lrp_dict = {int(key.split('.')[0].split('_')[-1]): (d['inp'], d['out']) for key, d in total_dict.items()}
    #print(lrp_dict)
    lists = sorted(lrp_dict.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples
    src = [el[0] for el in y]
    trg = [el[1] for el in y]
    src_mean = [numpy.mean(el) for el in src]
    src_se = [numpy.std(el)/numpy.sqrt(len(src)) for el in src]
    trg_mean = [numpy.mean(el) for el in trg]
    trg_se = [numpy.std(el)/numpy.sqrt(len(trg)) for el in trg]
    
    ax2 = axs[i*2+1][0]
    ax2.errorbar(x, src_mean, src_se, capsize=4, linestyle='-', marker='.', label='source')
    ax2.errorbar(x, trg_mean, trg_se, capsize=4, linestyle='-', marker='.', label='target prefix')
    ax2.legend()
    ax2.set_title(name + ' LRP', loc='center', wrap=True)
    ax2.grid(True)
    ax2.set_xlabel("# steps")
    ax2.set_ylabel("contribution")
    ax2.set_yticks(numpy.arange(0, 1., 0.1))
    
    ax3 = axs[i*2+1][1]
    ax3.errorbar(x, src_mean, src_se, capsize=4, linestyle='-', marker='.', label='source')
    ax3.errorbar(x, trg_mean, trg_se, capsize=4, linestyle='-', marker='.', label='target prefix')
    ax3.legend()
    ax3.set_title('LRP mean source and target prefix contributions over the course of training', loc='center', wrap=True)
    ax3.grid(True)
    ax3.set_xlabel("# steps")
    ax3.set_ylabel("contribution")
    ax3.set_yticks(numpy.arange(0, 1., 0.1))
    ax3.set_xscale('log')
    ax3.set_title(name + ' LRP (log)', loc='center', wrap=True)
    ticks = 10**numpy.arange(2,6)
    ax3.set_xticks(ticks, ticks)

fig.set_size_inches(8,8)
fig.savefig(f'plt/alti_vs_lrp.pdf')

