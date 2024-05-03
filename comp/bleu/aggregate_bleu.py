import os
import matplotlib.pyplot as plt
from sacrebleu.metrics import BLEU#, CHRF, TER
rootdir = '/cluster/scratch/ggabriel/ma/tm_val/evaluation_generate/'

bleu = BLEU(effective_order=True)
bleu_list = {}
halu_list = {}
for subdir, dirs, files in os.walk(rootdir):
    ckpt = subdir.split('/')[-1]
    if not ckpt:
        continue
    print(ckpt)
    f = os.path.join(subdir, 'generate-test.txt')
    halu_count = 0
    for line in open(f):
        if line[0] == 'T':
            ref = line.split('\t')[-1]
        elif line[0] == 'H':
            sys = line.split('\t')[-1]
            bleu_score = bleu.sentence_score(sys, [ref])
            #print(bleu_score)
            if bleu_score.score < 4:
                #print(sys, ref)
                halu_count += 1
    last_line = line
    bleu_score = float(last_line.split(' ')[6][:-1])
    bleu_list[int(ckpt)] = bleu_score
    halu_list[int(ckpt)] = halu_count
    print(halu_list)
print(halu_count)
import pickle
with open('bleu.pkl', 'wb') as f:
    pickle.dump(bleu_list, f)
lists = sorted(bleu_list.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.savefig('bleu.pdf')

plt.xscale('log')
plt.plot(x,y)
plt.savefig('bleu_log.pdf')

plt.clf()
with open('halu.pkl', 'wb') as f:
    pickle.dump(halu_list, f)
lists = sorted(halu_list.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.savefig('halu.pdf')

plt.yscale('log')
plt.savefig('halu_log.pdf')

