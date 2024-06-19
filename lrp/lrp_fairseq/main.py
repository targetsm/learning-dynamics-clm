import os
import torch

# Select GPU
#torch.cuda.set_device(7)
#torch.cuda.current_device()

import warnings
from pathlib import Path
from wrappers.transformer_wrapper import FairseqTransformerHub

import alignment.align as align
from wrappers.utils import visualize_alti
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
import random
import logging
logger = logging.getLogger()

logger.setLevel('WARNING')
warnings.simplefilter('ignore')

from dotenv import load_dotenv
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

data_sample = 'generate' # generate/interactive
teacher_forcing = True # teacher forcing/free decoding

green_color = '#82B366'
red_color = '#B85450'

# TODO: Code for itarting through the checkpoitns here
#       Also add lists for results storing

alti_dict = dict()
#directory = "/cluster/scratch/ggabriel/ma/tm/checkpoints/"
#directory = "../../models/tm/checkpoints/analysis"
directory = "./small_model/checkpoints/"
dir_out = './' # set the directory to save the results
result = {}

#for f in ["100.pt"]:
for f in ['checkpoint_last.pt']: #["100000.pt"]: #os.listdir(os.fsencode(directory)):
    
    filename = os.fsdecode(f)
    print(filename)
    result[filename] = []
    
    hub = FairseqTransformerHub.from_pretrained(
        checkpoint_dir=directory,
        checkpoint_file=filename,
        data_name_or_path="../data-bin/iwslt14.sep.tokenized.de-en/",
        #data_name_or_path="../../data-bin/iwslt14.sep.tokenized.de-en/",

        )
    #hub.models[0].to('cuda')
    # Get sample from provided test data
    total_source_contribution = 0
    total_target_contribution = 0
    len_testset_orig = len(open('./sentp_data/test.sentencepiece.de').readlines())
    len_testset = 1
    for i in range(len_testset):
        if data_sample == 'generate':
            # index in dataset
            i = 27
            
            sample = hub.get_sample('test', i)
            
            src_tensor = sample['src_tensor']
            tgt_tensor = sample['tgt_tensor']
            
            src_tok = sample['src_tok']
            tgt_tok = sample['tgt_tok']
            source_sentence = sample['src_tok']
            target_sentence = sample['tgt_tok']
        
        if data_sample == 'interactive':
            # index in dataset
            # i = 27 # index in dataset
            test_set_dir = "./sentp_data/"
            src = "de"
            tgt = "en"
            tokenizer = "sentencepiece"
            sample = hub.get_interactive_sample(i, test_set_dir, src,
                                                tgt, tokenizer, hallucination=None)
            src_tensor = sample['src_tensor']
            tgt_tensor = sample['tgt_tensor']
            source_sentence = sample['src_tok']
            target_sentence = sample['tgt_tok']


        if teacher_forcing:
            model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)
        
            #print("\n\nGREEDY DECODING\n")
            pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
            predicted_sentence = hub.decode(pred_tensor, hub.task.tgt_dict)
            pred_sent = hub.decode(pred_tensor, hub.task.tgt_dict, as_string=True)
            #print(f"Predicted sentence: \t {pred_sent}")
        #print(source_sentence)
        R_ = torch.zeros(log_probs.shape)
        inp_lrp = []
        out_lrp = []
        for i in range(len(predicted_sentence)):
            R_ = torch.zeros([1] + list(log_probs.shape)) #shape same as original lrp
            R_[0, i,pred_tensor[i]] = 1
            R = hub.relprop_ffn(R_, ("models.0.decoder.output_projection", "self.models[0].decoder.output_projection")) #model.loss._rdo_to_logits.relprop(R_)
            print(R)
            R = hub.relprop_decode(R)
            R_inp = torch.sum(torch.abs(hub.relprop_encode(R['enc_out'])), dim=-1)
            R_out = torch.sum(torch.abs(R['emb_out']), dim=-1)
            R_out_uncrop = torch.sum(torch.abs(R['emb_out_before_crop']), dim=-1)
            inp_lrp.append(R_inp[0])
            out_lrp.append(R_out_uncrop[0])        
            torch.cuda.empty_cache()
            print(R_inp[0], torch.sum(R_inp), R_out_uncrop[0], torch.sum(R_out_uncrop))
            exit()
        result[filename].append({'src': source_sentence, 'dst': target_sentence,
                   'inp_lrp': np.array(inp_lrp), 'out_lrp': np.array(out_lrp)})
        print(result[filename], 'inp', torch.sum(torch.stack(inp_lrp), -1), 'out',  torch.sum(torch.stack(out_lrp), -1), torch.mean(torch.sum(torch.stack(inp_lrp), -1)), torch.mean(torch.sum(torch.stack(out_lrp), -1)))
        exit()
import pickle
pickle.dump(result, open(dir_out + 'lrp_results', 'wb'))
