from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
from heapq import nsmallest
def cosine_similarity(x, y):
    return (x * y).sum(axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))


dir = "/local/home/ggabriel/ma/models/tm/evaluation_generate/"
file_dict = {}
for root, dirs, files in os.walk(dir):
    for name in files:
        if '.txt' in name:
            file_dict[int(root.split('/')[-1])] = os.path.join(root, name)

model = SentenceTransformer('sentence-transformers/LaBSE')


for key in file_dict.keys():
    try:
        with open('labse.pkl', 'rb') as f:
            similarity_dict = pickle.load(f)
    except:
        similarity_dict = {}

    print(key)
    lines = open(file_dict[key], 'r').readlines()
    targets = [x.split('\t')[-1][:-1].replace(' ', '').replace('▁', ' ') for x in lines if x[0] == 'T']
    hypos = [x.split('\t')[-1][:-1].replace(' ', '').replace('▁', ' ') for x in lines if x[0] == 'H']

        
    hypo_emb = model.encode(hypos)
    target_emb = model.encode(targets)
    similarities = cosine_similarity(hypo_emb, target_emb)
    similarity_dict[key] = (np.mean(similarities), np.std(similarities), np.mean(nsmallest(100, similarities)))
    print(sorted(similarity_dict.items(), key=lambda x: x[0]))
    with open('labse.pkl', 'wb') as f:
        pickle.dump(similarity_dict, f)
