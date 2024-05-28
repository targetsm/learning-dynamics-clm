import pickle
import numpy as np
import torch
def avg_lrp_by_pos(data, seg='inp'):
    count = 0
    res = np.zeros(data[0][seg + '_lrp'].shape[0])
    for i in range(len(data)):
        res += np.sum(data[i][seg + '_lrp'], axis=-1)
        count += 1
    res /= count
    return res


from scipy.stats import entropy

def all_inp_entropy(data, pos=None):
    res = []
    for i in range(len(data)):
        if not any(np.isnan(data[i]['inp_lrp'])):
            res_ = np.sum(data[i]['inp_lrp'], axis=-1)
            try:
                if pos is None:
                    res += [entropy(data[i]['inp_lrp'][p]/res_[p]) for p in range(data[i]['inp_lrp'].shape[0])]
                else:
                    res.append(entropy(data[i]['inp_lrp'][pos] / res_[pos]))
            except Exception:
                pass
    return res

def all_out_entropy(data, pos):
    res = []
    for i in range(len(data)):
        if not any(np.isnan(data[i]['out_lrp'])):
            res_ = np.sum(data[i]['out_lrp'], axis=-1)
            res.append(entropy(data[i]['out_lrp'][pos][:pos + 1] / res_[pos]))
    return res

def avg_lrp_by_src_pos_normed(data, ignore_eos=False):
    count = 0
    tgt_tokens = data[0]['inp_lrp'].shape[0]
    if ignore_eos:
        res = np.zeros(data[0]['inp_lrp'].shape[1] - 1)
    else:
        res = np.zeros(data[0]['inp_lrp'].shape[1])
    for i in range(len(data)):
        if not any(np.isnan(data[i]['inp_lrp'])):
            if ignore_eos:
                elem = data[i]['inp_lrp'][:, :-1] / np.sum(data[i]['inp_lrp'][:, :-1], axis=1).reshape([tgt_tokens, 1])
            else:
                elem = data[i]['inp_lrp'] / np.sum(data[i]['inp_lrp'], axis=1).reshape([tgt_tokens, 1])
            res += np.sum(elem, axis=0)
            count += 1
    res /= count
    res /= tgt_tokens 
    if ignore_eos:
        res *= (data[0]['inp_lrp'].shape[1] - 1)
    else:
        res *= data[0]['inp_lrp'].shape[1]
    return res

pictdir = './' # out dir, where to save the figures
dir_res = './' # dir with the LRP results
fname = 'lrp_results' # fname with the LRP results
data = pickle.load(open(dir_res + fname, 'rb'))
data = data['checkpoint_1_100.pt']
data[0]['inp_lrp'] = np.stack(data[0]['inp_lrp'])
data[0]['out_lrp'] = np.stack(data[0]['out_lrp'])

print(data)
print(len(data[0]['src']), len(data[0]['dst']), data[0]['inp_lrp'].shape, data[0]['out_lrp'].shape)
res1 = avg_lrp_by_pos(data, seg='inp')[:]
print(res1)
res2 = avg_lrp_by_pos(data, seg='out')[:]
print(res2)
print(res1 + res2)
