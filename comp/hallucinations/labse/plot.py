import matplotlib.pylab as plt
import pickle
import numpy
import sys

directory = sys.argv[1]

fig, ax = plt.subplots(nrows=1, sharex=True)

ax6 = ax
with open(f'{directory}/labse.pkl', 'rb') as f:
    labse_dict = pickle.load(f)
labse_lowest = {int(ckpt):x[2] for ckpt, x in labse_dict.items() if int(ckpt)}
labse_dict = {int(ckpt):(x[0],x[1]) for ckpt, x in labse_dict.items() if int(ckpt)}
lists = sorted(labse_dict.items()) # sorted by key, return a list of tuples
x, y_l = zip(*lists) # unpack a list of pairs into two tuples
y, y_v = zip(*y_l)

ax6.errorbar(x, y, y_v, capsize=4, linestyle='-', marker='.', label='LaBSE cos_sim')
#ax6.plot(x, y, linestyle='-', marker='.', label='laBSE cos_sim')
#lists = sorted(labse_lowest.items())   
#x, y = zip(*lists) # unpack a list of pairs into two tuples
#ax6.plot(x, y, linestyle='-', marker='.', label='laBSE low 100')
ax6.set_ylabel("Similarity")
ax6.grid()
ax6.legend()

fig.suptitle('Evolution of mean LaBSE similarity during Training', wrap=True)

#fig.set_size_inches(7,10)
plt.savefig(f'{directory}/labse.pdf')

plt.xscale('log')
fig.suptitle('Evolution of mean LaBSE similarity during Training (lin-log)', wrap=True)
ticks = [100, 500, 1000, 5000, 10000, 50000, 100000]
#ticks = [10000, 50000, 60000, 100000]
plt.xticks(ticks, ticks)
plt.savefig(f'{directory}/labse_log.pdf')

