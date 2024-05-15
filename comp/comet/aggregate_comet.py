import os
import matplotlib.pyplot as plt
from sacrebleu.metrics import BLEU#, CHRF, TER
rootdir = '/cluster/scratch/ggabriel/ma/tm_val/evaluation_generate/'

from comet import download_model, load_from_checkpoint

# Choose your model from Hugging Face Hub
#model_path = download_model("Unbabel/XCOMET-XL")
# or for example:
model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
model = load_from_checkpoint(model_path)
ckpt_dict = dict()
for subdir, dirs, files in os.walk(rootdir):
    ckpt = subdir.split('/')[-1]
    if not ckpt:
        continue
    print(ckpt)
    data_list = []
    f = os.path.join(subdir, 'generate-test.txt')
    for line in open(f):
        if line[0] == 'S':
            src = line.split('\t')[-1]
        if line[0] == 'T':
            ref = line.split('\t')[-1]
        elif line[0] == 'H':
            sys = line.split('\t')[-1]
            data_list.append({'src': src.replace(' ', '').replace('_', ' '), 'mt': sys.replace(' ', '').replace('_', ' '), 'ref':ref.replace(' ', '').replace('_', ' ')})
            if len(data_list) == 100:
                break
    model_output = model.predict(data_list,  batch_size=16, num_workers=1)
    ckpt_dict[ckpt] = model_output
    print(model_output)

import pickle
with open('comet.pkl', 'wb') as f:
    pickle.dump(ckpt_dict, f)
