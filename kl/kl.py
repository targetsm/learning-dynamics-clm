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

with open('unigram.pkl', 'rb') as f:
    unigram = pickle.load(f)
with open('bigram.pkl', 'rb') as f:
    vocab_idx, inv_idx, bigram = pickle.load(f)

print(file_list.keys())
kl_dict = dict()
kl_dict_unigram = dict()
kl_dict_bigram = dict()
for ckpt in file_list:
    print(ckpt)
    el = file_list[ckpt][0]
    tm_pd = pd.read_csv(f'/cluster/home/ggabriel/ma/kl/scores/tm/{el}', sep="\t", header=None)
    el = file_list[ckpt][1]
    lm_pd = pd.read_csv(f'/cluster/home/ggabriel/ma/kl/scores/lm_trunc/{el}', sep="\t", header=None)
    if lm_pd.empty or tm_pd.empty:
        continue

    kl = 0
    kl_unigram = 0
    kl_bigram = 0
    f_lm = open(f'/cluster/scratch/ggabriel/ma/lm_trunc/evaluation/{file_list[ckpt][1]}/probs-test.npy', 'rb')
    f_tm = open(f'/cluster/scratch/ggabriel/ma/tm/evaluation/{file_list[ckpt][0]}/probs-test.npy', 'rb')
    tm_vals = []
    lm_vals = []
    n_samples = lm_pd.shape[0]
    for i in range(n_samples):
        tm_vals.append(np.load(f_tm))
        lm_vals.append(np.load(f_lm))
    tm_pd[5] = tm_vals
    lm_pd[5] = lm_vals

    lm_pd.set_index(0, inplace=True)
    lm_pd.sort_index(inplace=True)
    tm_pd.set_index(0, inplace=True)
    tm_pd.sort_index(inplace=True)
    for i in range(n_samples):
        kl_loc = 0
        kl_loc_unigram = 0
        kl_loc_bigram = 0
        tm_scores = tm_pd.loc[i][5]
        lm_scores = lm_pd.loc[i][5]
        tokens = (tm_pd.loc[i][2]).split(' ')
        #print(tokens)
        length = len(lm_scores)
        #tm_scores = [float(idx) for idx in tm_pd[4][i].split(' ')]
        for j in range(length):
            kl_loc += np.sum(np.multiply(np.exp(lm_scores[j]),(lm_scores[j] - tm_scores[j])))
            #print(j, tokens, len(tokens), len(lm_scores))
            #smooth = 0.000000001
            #if j == len(tokens):
            #    unigram_prob = unigram['</s>']
            #else:
            unigram_prob = np.array(list(unigram.values()))
            #print('unigram_prob:', unigram_prob)
            #unigram_prob += smooth
            kl_loc_unigram += np.sum(np.multiply(unigram_prob,(np.log(unigram_prob) - tm_scores[j])))
            #print('kl_loc_unigram:', kl_loc_unigram)
            if j == 0:
                #bigram_prob = bigram[inv_idx['</s>'],inv_idx[tokens[j]]]
                bigram_prob = bigram[inv_idx['</s>'], :]
            elif j == len(tokens):
                #bigram_prob = bigram[inv_idx[tokens[j-1]],inv_idx['</s>']]
                bigram_prob = bigram[inv_idx[tokens[j-1]],:]
            else:
                #bigram_prob = bigram[inv_idx[tokens[j-1]],inv_idx[tokens[j]]]
                bigram_prob = bigram[inv_idx[tokens[j-1]],:]
            #print(sum(bigram_prob))
            #print('bigram_prob:', bigram_prob)
            #bigram_prob += smooth
            kl_loc_bigram += np.sum(np.multiply(bigram_prob,(np.log(bigram_prob) - tm_scores[j])))
            #print('kl_loc_bigram:', kl_loc_bigram)
        kl += kl_loc / length
        kl_unigram += kl_loc_unigram / length
        kl_bigram += kl_loc_bigram / length
        
    kl = kl / n_samples
    kl_unigram = kl_unigram  / n_samples
    kl_bigram = kl_bigram  / n_samples

    print(ckpt, 'kl-divergence:', kl)
    kl_dict[ckpt] = kl
    print(ckpt, 'kl-divergence unigram:', kl_unigram)
    kl_dict_unigram[ckpt] = kl_unigram
    print(ckpt, 'kl-divergence bigram:', kl_bigram)
    kl_dict_bigram[ckpt] = kl_bigram

kl_dict = {int(k):v for k,v in kl_dict.items()}
kl_dict_unigram = {int(k):v for k,v in kl_dict_unigram.items()}
kl_dict_bigram = {int(k):v for k,v in kl_dict_bigram.items()}

with open('kl_dict.pkl', 'wb') as f:
    pickle.dump(kl_dict, f)

with open('kl_dict_unigram.pkl', 'wb') as f:
    pickle.dump(kl_dict_unigram, f)

with open('kl_dict_bigram.pkl', 'wb') as f:
    pickle.dump(kl_dict_bigram, f)
    
import matplotlib.pylab as plt
lists = sorted(kl_dict.items()) # sorted by key, return a list of tuples
print(lists)
x, y = zip(*lists) # unpack a list of pairs into two tuples
print(x)
plt.plot(x,y)

lists = sorted(kl_dict_unigram.items()) # sorted by key, return a list of tuples
print(lists)
x, y = zip(*lists) # unpack a list of pairs into two tuples
print(x)
plt.plot(x,y)

lists = sorted(kl_dict_bigram.items()) # sorted by key, return a list of tuples
print(lists)
x, y = zip(*lists) # unpack a list of pairs into two tuples
print(x)
plt.plot(x,y)

plt.savefig('kl.png')
plt.xscale('log')
plt.plot(x, y)
plt.savefig('kl_log.png')
