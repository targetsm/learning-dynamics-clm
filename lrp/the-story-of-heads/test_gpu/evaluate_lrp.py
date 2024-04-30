import sys

sys.path.insert(0, '../') # insert your local path to the repo

import pickle
import numpy as np

VOC_PATH = './build_cpu/'

inp_voc = pickle.load(open(VOC_PATH + 'src.voc', 'rb'))
out_voc = pickle.load(open(VOC_PATH + 'dst.voc', 'rb'))


import tensorflow as tf
import lib
import lib.task.seq2seq.models.transformer_lrp as tr

tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

hp = {
     "num_layers": 4,
     "num_heads": 4,
     "ff_size": 128,
     "ffn_type": "conv_relu",
     "hid_size": 128,
     "emb_size": 128,
     "res_steps": "nlda",

     "rescale_emb": True,
     "inp_emb_bias": True,
     "normalize_out": True,
     "share_emb": False,
     "replace": 0,

     "relu_dropout": 0.1,
     "res_dropout": 0.1,
     "attn_dropout": 0.1,
     "label_smoothing": 0.1,

     "translator": "ingraph",
     "beam_size": 1,
     "beam_spread": 3,
     "len_alpha": 0.6,
     "attn_beta": 0,
}

model = tr.Model('mod', inp_voc, out_voc, inference_mode='fast', **hp)

path_to_ckpt = './build/checkpoint/model-latest.npz'
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
lib.train.saveload.load(path_to_ckpt, var_list)
datadir = '../data/'

print(open(datadir + 'test.de').readlines()[27:28])
print(model.translate_lines(open(datadir + 'test.de').readlines()[27:28]))

datadir = '../data/'

test_src = open(datadir + 'test.de').readlines()[27:28]
test_dst = open(datadir + 'test.en').readlines()[27:28]

feed_dict = model.make_feed_dict(zip(test_src[:1], test_dst[:1]))
ph = lib.task.seq2seq.data.make_batch_placeholder(feed_dict)
feed = {ph[key]: feed_dict[key] for key in feed_dict}

from lib.ops.record_activations import recording_activations
from lib.layers.basic import dropout_scope
from lib.ops import record_activations as rec
from lib.layers.lrp import LRP

def get_topk_logits_selector(logits, k=3):
    """ takes logits[batch, nout, voc_size] and returns a mask with ones at k largest logits """
    topk_logit_indices = tf.nn.top_k(logits, k=k).indices
    indices = tf.stack([
        tf.range(tf.shape(logits)[0] * tf.shape(logits)[1] * k) // (tf.shape(logits)[1] * k),
        (tf.range(tf.shape(logits)[0] * tf.shape(logits)[1] * k) // k) % tf.shape(logits)[1],
        tf.reshape(topk_logit_indices, [-1])
    ], axis=1)
    ones = tf.ones(shape=(tf.shape(indices)[0],))
    return tf.scatter_nd(indices, ones, shape=tf.shape(logits))

target_position = tf.placeholder(tf.int32, [])
with rec.recording_activations() as saved_activations, dropout_scope(False):

    rdo = model.encode_decode(ph, is_train=False)
    logits = model.loss._rdo_to_logits(rdo)
    out_mask = tf.one_hot(target_position, depth=tf.shape(logits)[1])[None, :, None]

    top1_logit = get_topk_logits_selector(logits, k=1) * tf.nn.softmax(logits)
    top1_prob = tf.reduce_sum(top1_logit, axis=-1)[0]

    R_ = get_topk_logits_selector(logits, k=1) * out_mask
    R = model.loss._rdo_to_logits.relprop(R_)
    R = model.transformer.relprop_decode(R)
    
    R_out = tf.reduce_sum(abs(R['emb_out']), axis=-1)
    R_inp = tf.reduce_sum(abs(model.transformer.relprop_encode(R['enc_out'])), axis=-1)
    R_out_uncrop = tf.reduce_sum(abs(R['emb_out_before_crop']), axis=-1)

dir_out = './' # set the directory to save the results
result = []
for elem in zip(test_src, test_dst):
    #print(len(result))
    src = elem[0].strip()
    dst = elem[1].strip()
    dst_words = len(dst.split()) + 1
    feed_dict = model.make_feed_dict(zip([src], [dst]))
    feed = {ph[key]: feed_dict[key] for key in feed_dict}
    
    inp_lrp = []
    out_lrp = []
    for token_pos in range(feed_dict['out'].shape[1]):
        print(elem, token_pos, feed)
        feed[target_position] = token_pos
        res_tot, res_inp, res_out, r_uncrop = sess.run((R, R_inp, R_out, R_out_uncrop), feed)
        print(res_tot['emb_out'].shape, res_tot['enc_out'].shape)
        print(res_out, np.array(res_out[0]).sum(-1))
        print(r_uncrop.shape, r_uncrop, np.array(r_uncrop[0]).sum(-1))
        inp_lrp.append(res_inp[0])
        out_lrp.append(r_uncrop[0])
    result.append({'src': src, 'dst': dst,
                   'inp_lrp': np.array(inp_lrp), 'out_lrp': np.array(out_lrp)
                  })
    
pickle.dump(result, open(dir_out + 'lrp_results', 'wb'))
