import numpy as np
import pickle

#vocab_en = open('/cluster/home/ggabriel/ma/data/sentp/iwslt14.sep.tokenized.de-en/code.en.vocab', 'r').readlines()
vocab_en = open('/local/home/ggabriel/ma/models/tl/wmt22frde/tm/data-bin/wmt22.sep.tokenized.fr-de/dict.de.txt', 'r').readlines()
vocab_en.insert(0, "<pad> 0")
vocab_en.insert(1, "</s> 0")
vocab_en.insert(2, "<unk> 0")
vocab_en.insert(3, "<mask> 0")
vocab_idx = [x.split(' ')[0] for x in vocab_en]
inv_idx = {x:vocab_idx.index(x) for x in vocab_idx}
unigram_en = {x.split(' ')[0]:0 for x in vocab_en}
vocab_size = len(vocab_idx)
bigram_en = np.zeros((vocab_size, vocab_size), np.int)

print(vocab_idx)
c = 0
with open('/local/home/ggabriel/ma/data/tl/wmt22frde/wmt22.sep.tokenized.fr-de/train.de', 'r') as f_en:
    for line in f_en:
        tokens = [ '</s>'] + line[:-1].split(' ') + ['</s>']
        for i in range(1, len(tokens)):
            t = tokens[i]
            t_prev = tokens[i-1]
            if not t in unigram_en.keys():
                unigram_en['<unk>'] += 1
            else:
                unigram_en[t]+=1
            if t not in inv_idx.keys():
                t = '<unk>'
            if t_prev not in inv_idx.keys():
                t_prev = '<unk>'
            bigram_en[inv_idx[t_prev], inv_idx[t]] += 1



#unigram
smooth = 0.000000001
unigram_dict = {key:(value/sum(unigram_en.values()))+smooth for (key,value) in unigram_en.items()} 
with open('unigram.pkl', 'wb') as f:
    pickle.dump(unigram_dict, f)

# bigram
col_sums = bigram_en.sum(axis=1, keepdims=True)
bigram_normalized = bigram_en / col_sums
bigram_normalized[np.isnan(bigram_normalized)] = 0
bigram_normalized += smooth
with open('bigram.pkl', 'wb') as f:
    pickle.dump((vocab_idx, inv_idx, bigram_normalized), f)
