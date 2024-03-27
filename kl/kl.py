import pandas as pd
import math
import os

from os import listdir
from os.path import join, isfile

path = "/cluster/home/ggabriel/ma/kl/scores"
file_list = dict()
for filename in listdir(path):  # iterates over all the files in 'path'
    import re
    print(filename)
    if not re.match(r"checkpoint_[0-9]+_[0-9]+.[A-Za-z]+", filename):
        continue
    from pathlib import Path
    stem = Path(filename).stem.split('_')[-1]
    if not stem in file_list:
        file_list[stem]=list()
    file_list[stem].append(Path(filename))

print(file_list)
kl_dict = dict()
for ckpt in file_list:
    try:
        for el in file_list[ckpt]:
            if 'tm' in str(el):
                tm_pd = pd.read_csv(f'scores/{el}', sep="\t", header=None)
            else:
                lm_pd = pd.read_csv(f'scores/{el}', sep="\t", header=None)
    except:
        continue
    if lm_pd.empty:
        continue
    lm_pd.set_index(0, inplace=True)
    lm_pd.sort_index(inplace=True)
    tm_pd.set_index(0, inplace=True)
    tm_pd.sort_index(inplace=True)

    kl = 0
    for i in range(lm_pd.shape[0]):
        kl_loc = 0
        lm_scores = [float(idx) for idx in lm_pd[4][i].split(' ')]
        tm_scores = [float(idx) for idx in tm_pd[4][i].split(' ')]
        for j in range(len(lm_scores)):
            kl_loc += math.exp(lm_scores[j]) * (lm_scores[j] - tm_scores[j])
            norm_1 = (1-math.exp(lm_scores[j]))/10000
            norm_2 = (1-math.exp(tm_scores[j]))/10000
            kl_loc += 9999*(norm_1 * math.log(norm_1/norm_2))
        kl += kl_loc / len(lm_scores)

    kl = kl / lm_pd.shape[0]
    print(ckpt, 'kl-divergence:', kl)
    kl_dict[ckpt] = kl

kl_dict = {int(k):v for k,v in kl_dict.items()}
import matplotlib.pylab as plt
lists = sorted(kl_dict.items()) # sorted by key, return a list of tuples
print(lists)
x, y = zip(*lists) # unpack a list of pairs into two tuples
print(x)
plt.plot(x,y)
plt.savefig('kl.png')
plt.xscale('log')
plt.plot(x, y)
plt.savefig('kl_log.png')
