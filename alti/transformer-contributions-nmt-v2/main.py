import os
import torch

# Select GPU
#torch.cuda.set_device(0)
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


import pickle
try:
    with open('alti_results_iwsltxiwslt.pkl', 'rb') as f:
        alti_dict = pickle.load(f)
except:
    alti_dict = dict()
directory = "/local/home/ggabriel/ma/models/tl/iwslt14deen/tm/checkpoints/"
for f in os.listdir(os.fsencode(directory)):
    filename = os.fsdecode(f)
    print(filename)
    if 'best' in filename or 'last' in filename or 'analysis' in filename or int(filename.split('_')[-1][:-3]) in alti_dict:
        continue
    
    hub = FairseqTransformerHub.from_pretrained(
        checkpoint_dir=directory,
        checkpoint_file=filename,
        data_name_or_path="../data-bin/iwslt14.sep.tokenized.de-en/",
        )
    
    # Get sample from provided test data
    total_source_contribution = 0
    total_target_contribution = 0
    step_number = int(filename.split('_')[-1][:-3])
    alti_dict[step_number]={'src':[], 'trg':[]}
    #len_testset_orig = len(open('./sentp_data/test.sentencepiece.de').readlines())
    len_testset = min(1000,  len(open("../../data/tl/iwslt14deen/sentp/iwslt14.sep.tokenized.de-en/test.en").readlines()))
    print(len_testset)
    for i in range(len_testset):
        if data_sample == 'generate':
            # index in dataset
            # i = 29
            
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
            test_set_dir = "../../data/tl/wmt22frde/wmt22.sep.tokenized.fr-de"
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
        
        
        if not teacher_forcing:
            tgt_tensor_free = []
            # Add tokens to prefix_tokens to force decoding with initial tokens
            # We use token id = 3 (unk) to generate hallucinations
            # prefix_tokens = torch.tensor([[3]]).to('cuda')
            # inference_step_args={'prefix_tokens': prefix_tokens}
            inference_step_args = None
        
            #print("\n\nBEAM SEARCH\n")
            for pred in hub.generate(src_tensor, beam=4, inference_step_args = inference_step_args):
                tgt_tensor_free.append(pred['tokens'])
                #pred_sent = hub.decode(pred['tokens'], hub.task.tgt_dict, as_string=True)
                #score = pred['score'].item()
                #print(f"{score} \t {pred_sent}")
        
            hy8po = 0 # first hypothesis
            tgt_tensor = tgt_tensor_free[hypo]
            
            # We add eos token at the beginning of sentence and delete it from the end
            tgt_tensor = torch.cat([torch.tensor([hub.task.target_dictionary.eos_index]).to(tgt_tensor.device),
                            tgt_tensor[:-1]
                        ]).to(tgt_tensor.device)
            #8target_sentence = hub.decode(tgt_tensor, hub.task.target_dictionary, as_string=False)
        
            # Forward-pass to get the 'prediction' (predicted_sentence) when the top-hypothesis is in the decoder input
            model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)
        
            #print(f"\n\nGREEDY DECODING with hypothesis {hypo+1}\n")
            pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
            predicted_sentence = hub.decode(pred_tensor, hub.task.target_dictionary)
            #pred_sent = hub.decode(pred_tensor, hub.task.target_dictionary, as_string=True)
            #print(f"Predicted sentence: \t {pred_sent}") # result should match beam search when beam=1
        total_alti = hub.get_contribution_rollout(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum')['total']
        #word_level = False
        #alignment = False # evaluating alignments, predictions (rows) are the reference, (not showing real interpretations)
        
        #alti_result, source_sentence_, predicted_sentence_ = visualize_alti(total_alti, source_sentence,
        #                                                                    [target_sentence[0]] + ['▁' + target_sentence[1]] + target_sentence[2:], predicted_sentence,
        #                                                                    word_level, alignment, all_layers = False)
        #print(alti_result[:, :len(source_sentence_)].sum(-1), alti_result[:, :len(source_sentence_)].sum(-1).sum()/len(predicted_sentence))
        #print(alti_result[:, len(source_sentence_):].sum(-1), alti_result[:, len(source_sentence_):].sum(-1).sum()/len(predicted_sentence))
        layer = -1
        contributions_rollout_layer = total_alti[layer]
        alti_result = contributions_rollout_layer.detach().cpu().numpy()
        alti_dict[step_number]['src'].append(alti_result[:, :len(source_sentence)].sum(-1).sum()/len(predicted_sentence))
        alti_dict[step_number]['trg'].append(alti_result[:, len(source_sentence):].sum(-1).sum()/len(predicted_sentence))
        if i % 10 == 0:
            torch.cuda.empty_cache()
        #print(total_source_contribution/(i+1))
        #print(total_target_contribution/(i+1))
    #total_source_contribution /= len_testset
    #total_target_contribution /= len_testset
    
    #print(alti_dict)
    print('source_contribution: ', np.mean(alti_dict[step_number]['src']), np.std(alti_dict[step_number]['src']))
    print('target_contribution: ', np.mean(alti_dict[step_number]['trg']), np.std(alti_dict[step_number]['trg']))
    step_number = int(filename.split('_')[-1][:-3])
    #alti_dict[step_n] = (total_source_contribution, total_target_contribution)
    with open('alti_results_iwsltxiwslt.pkl', 'wb') as f:
        pickle.dump(alti_dict, f)
# TODO: Add output logic here, save lists in a file
import pickle 

with open('alti_results_iwsltxiwslt.pkl', 'wb') as f:
    pickle.dump(alti_dict, f)
