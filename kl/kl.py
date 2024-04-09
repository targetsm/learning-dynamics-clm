import pandas as pd
import math
import os
import re
import numpy as np
import pickle
from pathlib import Path
from os import listdir
from os.path import join, isfile

path = "/cluster/home/ggabriel/ma/kl/scores/tm"
file_list = dict()
for filename in listdir(path):  # iterates over all the files in 'path'
    print(filename)
    if not re.match(r"checkpoint_[0-9]+_[0-9]+", filename):
        continue
    stem = Path(filename).stem.split('_')[-1]
    if not stem in file_list:
        file_list[stem]=list()
    file_list[stem].append(filename)

path = "/cluster/home/ggabriel/ma/kl/scores/lm_trunc"
for filename in listdir(path):  # iterates over all the files in 'path'
    print(filename)
    if not re.match(r"checkpoint_[0-9]+_[0-9]+", filename):
        continue
    stem = Path(filename).stem.split('_')[-1]
    if not stem in file_list:
        file_list[stem]=list()
    file_list[stem].append(filename)

print(file_list.keys())
kl_dict = dict()
for ckpt in file_list:
    el = file_list[ckpt][0]
    tm_pd = pd.read_csv(f'/cluster/home/ggabriel/ma/kl/scores/tm/{el}', sep="\t", header=None)
    el = file_list[ckpt][1]
    lm_pd = pd.read_csv(f'/cluster/home/ggabriel/ma/kl/scores/lm_trunc/{el}', sep="\t", header=None)
    if lm_pd.empty or tm_pd.empty:
        continue

    kl = 0
    f_lm = open(f'/cluster/scratch/ggabriel/ma/lm_trunc/evaluation/{file_list[ckpt][1]}/probs-test.npy', 'rb')
    f_tm = open(f'/cluster/scratch/ggabriel/ma/tm/evaluation/{file_list[ckpt][0]}/probs-test.npy', 'rb')
    tm_vals = []
    lm_vals = []
    for i in range(lm_pd.shape[0]):
        kl_loc = 0
        tm_vals.append(np.load(f_tm))
        lm_vals.append(np.load(f_lm))
    tm_pd[5] = tm_vals
    lm_pd[5] = lm_vals

    lm_pd.set_index(0, inplace=True)
    lm_pd.sort_index(inplace=True)
    tm_pd.set_index(0, inplace=True)
    tm_pd.sort_index(inplace=True)

    kl = 0
    for i in range(lm_pd.shape[0]):
        kl_loc = 0
        tm_scores = tm_pd.loc[i][5]
        lm_scores = lm_pd.loc[i][5]
        #tm_scores = [float(idx) for idx in tm_pd[4][i].split(' ')]
        for j in range(len(lm_scores)):
            kl_loc += np.sum(np.multiply(np.exp(lm_scores[j]),(lm_scores[j] - tm_scores[j])))
            #kl_loc = 0
            #for k in range(len(lm_scores[j])):
            #    kl_loc += math.exp(lm_scores[j][k]) * (lm_scores[j][k] - tm_scores[j][k])
            #print(kl_loc)
            #norm_1 = (1-math.exp(lm_scores[j]))/10000
            #norm_2 = (1-math.exp(tm_scores[j]))/10000
            #kl_loc += 9999*(norm_1 * math.log(norm_1/norm_2))
        kl += kl_loc / len(lm_scores)
    kl = kl / lm_pd.shape[0]
    print(ckpt, 'kl-divergence:', kl)
    kl_dict[ckpt] = kl

kl_dict = {int(k):v for k,v in kl_dict.items()}
with open('kl_dict.pkl', 'wb') as f:
    pickle.dump(kl_dict, f)
    
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
