import pickle
import numpy
import sys
import matplotlib.pylab as plt

fig, axs = plt.subplots(8, 1, sharex=False, constrained_layout=True)
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.08)

dir_list = ['iwslt14deen/iwslt/', 'iwslt14deen/wmt/', 'wmt22frde_subset/iwslt/', 'wmt22frde_subset/wmt/', 'wmt22frde/wmt/', 'wmt22frde/wmt_big/', 'wmt22deen_subset/iwslt/', 'wmt22deen_subset/wmt/']
    
for i in range(len(dir_list)):
    with open(f'plt/{dir_list[i]}/labse.pkl', 'rb') as f:
        labse_dict = pickle.load(f)
    labse_dict = {int(ckpt):(x[0],x[1]) for ckpt, x in labse_dict.items() if int(ckpt)}
    lists = sorted(labse_dict.items()) # sorted by key, return a list of tuples
    x, y_l = zip(*lists) # unpack a list of pairs into two tuples
    y, y_v = zip(*y_l)

    axs[i].errorbar(x, y, y_v, capsize=4, linestyle='-', marker='.', label='LaBSE cos_sim')
    axs[i].grid()
    axs[i].legend()
    axs[i].set_yticks(numpy.arange(0, 1., 0.1))

fig.suptitle('Evolution of mean LaBSE similarity during Training', wrap=True)
plt.xlabel('# steps')
plt.ylabel("Similarity")
fig.set_size_inches(10,20)
plt.savefig(f'plt/labse.pdf')

plt.xscale('log')
fig.suptitle('Evolution of mean LaBSE similarity during Training (lin-log)', wrap=True)
ticks = [100, 500, 1000, 5000, 10000, 50000, 100000]
#ticks = [10000, 50000, 60000, 100000]
plt.xticks(ticks, ticks)
plt.savefig(f'plt/labse_log.pdf')
