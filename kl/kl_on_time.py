import pandas as pd
import math
import os
import re
import numpy as np
import pickle
from pathlib import Path
from os import listdir
from os.path import join, isfile

import sys
ckpt = int(sys.argv[1])
print(ckpt)
with open('unigram.pkl', 'rb') as f:
    unigram = pickle.load(f)
with open('bigram.pkl', 'rb') as f:
    vocab_idx, inv_idx, bigram = pickle.load(f)


try:
    with open('kl_dict.pkl', 'rb') as f:
        kl_dict = pickle.load(f)

    with open('kl_dict_unigram.pkl', 'rb') as f:
        kl_dict_unigram = pickle.load(f)

    with open('kl_dict_bigram.pkl', 'rb') as f:
        kl_dict_bigram = pickle.load(f)
except:
    kl_dict = dict()
    kl_dict_unigram = dict()
    kl_dict_bigram = dict()

tm_pd = pd.read_csv(f'/cluster/scratch/ggabriel/ma/evaluation/scores_tm.txt', sep="\t", header=None)
lm_pd = pd.read_csv(f'/cluster/scratch/ggabriel/ma/evaluation/scores_lm.txt', sep="\t", header=None)
kl = 0
kl_unigram = 0
kl_bigram = 0
f_lm = open(f'/cluster/scratch/ggabriel/ma/evaluation/lm/probs-test.npy', 'rb')
f_tm = open(f'/cluster/scratch/ggabriel/ma/evaluation/tm/probs-test.npy', 'rb')
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
    length = len(lm_scores)
    for j in range(length):
        kl_loc += np.sum(np.multiply(np.exp(lm_scores[j]),(lm_scores[j] - tm_scores[j])))
        unigram_prob = np.array(list(unigram.values()))
        kl_loc_unigram += np.sum(np.multiply(unigram_prob,(np.log(unigram_prob) - tm_scores[j])))
        if j == 0:
            bigram_prob = bigram[inv_idx['</s>'], :]
        elif j == len(tokens):
            bigram_prob = bigram[inv_idx[tokens[j-1]],:]
        else:
            bigram_prob = bigram[inv_idx[tokens[j-1]],:]
        kl_loc_bigram += np.sum(np.multiply(bigram_prob,(np.log(bigram_prob) - tm_scores[j])))
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

#kl_dict = {int(k):v for k,v in kl_dict.items()}
#kl_dict_unigram = {int(k):v for k,v in kl_dict_unigram.items()}
#kl_dict_bigram = {int(k):v for k,v in kl_dict_bigram.items()}

with open('kl_dict.pkl', 'wb') as f:
    pickle.dump(kl_dict, f)

with open('kl_dict_unigram.pkl', 'wb') as f:
    pickle.dump(kl_dict_unigram, f)

with open('kl_dict_bigram.pkl', 'wb') as f:
    pickle.dump(kl_dict_bigram, f)
    