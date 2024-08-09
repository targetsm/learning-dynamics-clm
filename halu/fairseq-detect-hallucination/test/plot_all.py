import pickle
import numpy
import sys
import matplotlib.pylab as plt

fig, axs = plt.subplots(4, 2, sharex=False, constrained_layout=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.08)

dir_list = ['iwslt14deen/iwslt/', 'iwslt14deen/wmt/'] 

for i in range(len(dir_list)):
    with open(f'/local/home/ggabriel/ma/comp/hallucinations/labse/plt/{dir_list[i]}/labse.pkl', 'rb') as f:
        labse_dict = pickle.load(f)
    labse_dict = {int(ckpt):(x[0],x[1]) for ckpt, x in labse_dict.items() if int(ckpt)}
    lists = sorted(labse_dict.items()) # sorted by key, return a list of tuples
    x, y_l = zip(*lists) # unpack a list of pairs into two tuples
    y, y_v = zip(*y_l)
    name = dir_list[i].replace('/', '-').replace('_', '-').replace('wmt-big-', 'large').replace('wmt-', 'medium').replace('iwslt-', 'small')
    print(name)
    ax0 = axs[i*2][0]
    ax0.errorbar(x, y, y_v, capsize=4, linestyle='-', marker='.', label='LaBSE cos_sim')
    ax0.grid()
    ax0.legend()
    ax0.set_yticks(numpy.arange(0, 1., 0.1))
    ax0.set_title(name + ' LaBSE')
    ax0.set_xlabel('# steps')
    ax0.set_ylabel("similarity")
    
    ax1 = axs[i*2][1]
    ax1.set_xscale('log')
    ax1.errorbar(x, y, y_v, capsize=4, linestyle='-', marker='.', label='LaBSE cos_sim')
    ax1.grid()
    ax1.legend()
    ax1.set_yticks(numpy.arange(0, 1., 0.1))
    ax1.set_title(name + ' LaBSE')
    ax1.set_xlabel('# steps')
    ax1.set_ylabel("similarity")
    ax1.set_title(name + ' LaBSE (log)')
    ticks = 10**numpy.arange(2,6)
    #ticks = [10000, 50000, 60000, 100000]
    ax1.set_xticks(ticks, ticks)

    with open(f'plt/{dir_list[i]}/results.pkl', 'rb') as f:
        results_dict = pickle.load(f)

    lists = sorted(results_dict.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    halu_mean = [numpy.mean(j) for j in y]
    halu_se = [numpy.std(j)/numpy.sqrt(len(j)) for j in y]
    ax2 = axs[i*2+1][0]
    ax2.errorbar(x, halu_mean, halu_se, capsize=4, linestyle='-', marker='.', label='ratio')
    ax2.legend()
    ax2.set_title(name + ' token hal.')
    ax2.grid(True)
    ax2.set_xlabel("# steps")
    ax2.set_ylabel("ratio")
    #plt.yticks(numpy.arange(0, 1., 0.1))
    ax2.set_yticks(numpy.arange(0, 1., 0.1))
    ax3 = axs[i*2+1][1]
    ax3.errorbar(x, halu_mean, halu_se, capsize=4, linestyle='-', marker='.', label='ratio')
    ax3.legend()
    ax3.set_title(name + ' token hal. (log)')
    ax3.grid(True)
    ax3.set_xlabel("# steps")
    ax3.set_ylabel("ratio")
    #plt.yticks(numpy.arange(0, 1., 0.1))
    ax3.set_yticks(numpy.arange(0, 1., 0.1))

    ax3.set_xscale('log')
    ticks = 10**numpy.arange(2,6)
    ax3.set_xticks(ticks, ticks)
fig.set_size_inches(8,8)
fig.savefig(f'plt/lrp_vs_halu.pdf')
