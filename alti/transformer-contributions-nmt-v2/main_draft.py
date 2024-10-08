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

import logging
logger = logging.getLogger()

logger.setLevel('WARNING')
warnings.simplefilter('ignore')

from dotenv import load_dotenv
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

data_sample = 'generate' # generate/interactive
teacher_forcing = False # teacher forcing/free decoding

green_color = '#82B366'
red_color = '#B85450'


alti_dict = dict()
directory = "/cluster/scratch/ggabriel/ma/tm_val/checkpoints/analysis"
for f in os.listdir(os.fsencode(directory)):
    filename = os.fsdecode(f)
    print(filename)
    
    hub = FairseqTransformerHub.from_pretrained(
        checkpoint_dir=directory,
        checkpoint_file=filename,
        data_name_or_path="../../data-bin/iwslt14.sep.tokenized.de-en/",
        )
    
    # Get sample from provided test data
    total_source_contribution = 0
    total_target_contribution = 0
    len_testset = len(open('./sentp_data/test.sentencepiece.de').readlines())
    #len_testset = 10
    for i in range(len_testset):
        if data_sample == 'generate':
            # index in dataset
            # i = 29
            hub.task.load_dataset('test')

            # iterate over mini-batches of data
            batch_itr = hub.task.get_batch_iterator(
                hub.task.dataset('test'), max_tokens=4096,
            ).next_epoch_itr()
            for batch in batch_itr:
                print(batch)
                break
            src = batch['net_input']
            trg = batch['target']
            src_tensor = batch['net_input']['src_tokens']
            tgt_tensor = batch['target']
            print(src_tensor)
            print(tgt_tensor)
            #sample = hub.get_batch('test', 10)
            
            #src_tensor = sample['src_tensor']
            #tgt_tensor = sample['tgt_tensor']
            
            #src_tok = sample['src_tok']
            #tgt_tok = sample['tgt_tok']
            #source_sentence = sample['src_tok']
            #target_sentence = sample['tgt_tok']
        
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
        
        
        if not teacher_forcing:
            tgt_tensor_free = []
            # Add tokens to prefix_tokens to force decoding with initial tokens
            # We use token id = 3 (unk) to generate hallucinations
            # prefix_tokens = torch.tensor([[3]]).to('cuda')
            # inference_step_args={'prefix_tokens': prefix_tokens}
            inference_step_args = None
        
            #print("\n\nBEAM SEARCH\n")
            target_length = tgt_tensor.shape[1]
            for pred in hub.generate(src_tensor, beam=4, inference_step_args = inference_step_args):
                prediction = pred[0]['tokens']
                prediction_conv = torch.cat([torch.tensor([hub.task.target_dictionary.eos_index]).to(prediction.device),
                            prediction[:-1]]).to(prediction.device)
                prediction_conv = torch.cat((prediction_conv, torch.ones(target_length - len(prediction_conv))))
                print(prediction_conv.shape) 
                
                tgt_tensor_free.append(prediction_conv)
                #pred_sent = hub.decode(pred['tokens'], hub.task.tgt_dict, as_string=True)
                #score = pred['score'].item()
                #print(f"{score} \t {pred_sent}")
            tgt_tensor = torch.stack(tgt_tensor_free)
            print(tgt_tensor.shape)
            hypo = 0 # first hypothesis
            #tgt_tensor = tgt_tensor_free[hypo]
            
            # We add eos token at the beginning of sentence and delete it from the end
            #tgt_tensor = torch.cat([torch.tensor([hub.task.target_dictionary.eos_index]).to(tgt_tensor.device),
            #                tgt_tensor[:-1]
            #            ]).to(tgt_tensor.device)
            #target_sentence = hub.decode(tgt_tensor, hub.task.target_dictionary, as_string=False)
        
            # Forward-pass to get the 'prediction' (predicted_sentence) when the top-hypothesis is in the decoder input
            #model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)
        
            #print(f"\n\nGREEDY DECODING with hypothesis {hypo+1}\n")
            #pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
            #predicted_sentence = hub.decode(pred_tensor, hub.task.target_dictionary)
            #pred_sent = hub.decode(pred_tensor, hub.task.target_dictionary, as_string=True)
            #print(f"Predicted sentence: \t {pred_sent}") # result should match beam search when beam=1
        total_alti = hub.get_contribution_rollout(src, tgt_tensor, 'l1', norm_mode='min_sum')['total']
        #word_level = False
        #alignment = False # evaluating alignments, predictions (rows) are the reference, (not showing real interpretations)
        
        #alti_result, source_sentence_, predicted_sentence_ = visualize_alti(total_alti, source_sentence,
        #                                                                    [target_sentence[0]] + ['▁' + target_sentence[1]] + target_sentence[2:], predicted_sentence,
        #                                                                    word_level, alignment, all_layers = False)
        #print(alti_result[:, :len(source_sentence_)].sum(-1), alti_result[:, :len(source_sentence_)].sum(-1).sum()/len(predicted_sentence))
        #print(alti_result[:, len(source_sentence_):].sum(-1), alti_result[:, len(source_sentence_):].sum(-1).sum()/len(predicted_sentence))
        layer = -1
        #print(tota
        contributions_rollout_layer = total_alti[layer]
        alti_result = contributions_rollout_layer.detach().cpu().numpy()
        total_source_contribution+=alti_result[:, :len(source_sentence)].sum(-1).sum()/len(predicted_sentence)
        total_target_contribution+=alti_result[:, len(source_sentence):].sum(-1).sum()/len(predicted_sentence)
    
        #print(total_source_contribution/(i+1))
        #print(total_target_contribution/(i+1))
        
    total_source_contribution /= len_testset
    total_target_contribution /= len_testset
    
    print('source_contribution: ', total_source_contribution)
    print('target_contribution: ', total_target_contribution)
    step_number = int(filename.split('_')[-1][:-3])
    alti_dict[step_number] = (total_source_contribution, total_target_contribution)
    print(alti_dict)

import pickle 

with open('alti_results.pkl', 'wb') as f:
    pickle.dump(alti_dict, f)
