import matplotlib.pylab as plt
import pickle
import numpy


valid_list = [x.split('\t') for x in open('./plt/tl/iwslt14deen/valid_list.txt', 'r').readlines()]
valid_list = [(int(y), float(x)) for [x,y] in valid_list]
train_list = [x.split('\t') for x in open('./plt/tl/iwslt14deen/train_list.txt', 'r').readlines()]
train_list = [(int(y), float(x)) for [x,y] in train_list]

fig, ax = plt.subplots(nrows=2, sharex=True)

ax3 = ax[0]
ax3.plot(*zip(*train_list), linestyle='-', marker='.', label='train loss')
ax3.plot(*zip(*valid_list), linestyle='-', marker='.', label='val loss')
ax3.set_ylabel("loss")
ax3.grid()
ax3.legend()

valid_list2 = [x.split('\t') for x in open('./plt/tl/wmt22frde/valid_list.txt', 'r').readlines()]
valid_list2 = [(int(y), float(x)) for [x,y] in valid_list2]
train_list2 = [x.split('\t') for x in open('./plt/tl/wmt22frde/train_list.txt', 'r').readlines()]
train_list2 = [(int(y), float(x)) for [x,y] in train_list2]

ax1 = ax[1]
ax1.plot(*zip(*train_list2), linestyle='-', marker='.', label='train loss')
ax1.plot(*zip(*valid_list2), linestyle='-', marker='.', label='val loss')
ax1.set_ylabel("loss")
ax1.grid()
ax1.legend()

fig.suptitle('Training and Validation Loss Evolution during Training', wrap=True)

#fig.set_size_inches(7,10)
plt.savefig('train_val_loss.pdf')

plt.xscale('log')
fig.suptitle('Training and Validation Loss Evolution during Training (lin-log)', wrap=True)
ticks = [100, 500, 1000, 5000, 10000, 50000, 100000]
#ticks = [10000, 50000, 60000, 100000]
plt.xticks(ticks, ticks)
plt.savefig('train_val_loss_log.pdf')

