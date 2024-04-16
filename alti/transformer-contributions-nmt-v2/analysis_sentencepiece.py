import os
import torch

# Select GPU
#torch.cuda.set_device(5)
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

data_sample = 'interactive' # generate/interactive
teacher_forcing = False # teacher forcing/free decoding

green_color = '#82B366'
red_color = '#B85450'

hub = FairseqTransformerHub.from_pretrained(
    checkpoint_dir="/cluster/scratch/ggabriel/ma/tm/checkpoints",
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path="../data-bin/iwslt14.sep.tokenized.de-en/",
    #bpe='subword_nmt',
    #bpe_codes="../../data/iwslt14.tokenized.de-en/code"
    )

# Get sample from provided test data

if data_sample == 'generate':
    # index in dataset
    i = 29

    sample = hub.get_sample('test', i)

    src_tensor = sample['src_tensor']
    tgt_tensor = sample['tgt_tensor']
    src_tok = sample['src_tok']
    src_sent = sample['src_sent']
    tgt_sent = sample['tgt_sent']
    print(f"\nSource sentence: \t {src_sent}")
    print(f"Target sentence: \t {tgt_sent}")

if data_sample == 'interactive':
    # index in dataset
    i = 27 # index in dataset
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
    print(source_sentence, target_sentence)
    print(sample['src_word_sent'])
    print(sample['tgt_word_sent'])

if teacher_forcing:
    model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)

    print("\n\nGREEDY DECODING\n")
    pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
    predicted_sentence = hub.decode(pred_tensor, hub.task.tgt_dict)
    pred_sent = hub.decode(pred_tensor, hub.task.tgt_dict, as_string=True)
    print(f"Predicted sentence: \t {pred_sent}")


if not teacher_forcing:
    tgt_tensor_free = []
    # Add tokens to prefix_tokens to force decoding with initial tokens
    # We use token id = 3 (unk) to generate hallucinations
    # prefix_tokens = torch.tensor([[3]]).to('cuda')
    # inference_step_args={'prefix_tokens': prefix_tokens}
    inference_step_args = None

    print("\n\nBEAM SEARCH\n")
    for pred in hub.generate(src_tensor, beam=4, inference_step_args = inference_step_args):
        tgt_tensor_free.append(pred['tokens'])
        pred_sent = hub.decode(pred['tokens'], hub.task.tgt_dict, as_string=True)
        score = pred['score'].item()
        print(f"{score} \t {pred_sent}")

    hypo = 0 # first hypothesis
    tgt_tensor = tgt_tensor_free[hypo]
    
    # We add eos token at the beginning of sentence and delete it from the end
    tgt_tensor = torch.cat([torch.tensor([hub.task.target_dictionary.eos_index]).to(tgt_tensor.device),
                    tgt_tensor[:-1]
                ]).to(tgt_tensor.device)
    target_sentence = hub.decode(tgt_tensor, hub.task.target_dictionary, as_string=False)

    # Forward-pass to get the 'prediction' (predicted_sentence) when the top-hypothesis is in the decoder input
    model_output, log_probs, encoder_out, layer_inputs, layer_outputs = hub.trace_forward(src_tensor, tgt_tensor)

    print(f"\n\nGREEDY DECODING with hypothesis {hypo+1}\n")
    pred_log_probs, pred_tensor = torch.max(log_probs, dim=-1)
    predicted_sentence = hub.decode(pred_tensor, hub.task.target_dictionary)
    pred_sent = hub.decode(pred_tensor, hub.task.target_dictionary, as_string=True)
    print(f"Predicted sentence: \t {pred_sent}") # result should match beam search when beam=1

layer = 5

cross_attn_contributions = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, 'attn_w', norm_mode='sum_one')['decoder.encoder_attn'])
cross_attn_contributions = cross_attn_contributions.detach().cpu().numpy()
plt.figure(figsize=(6,6))
df = pd.DataFrame(cross_attn_contributions[layer],columns=source_sentence,index=predicted_sentence)
sns.set(font_scale=1.2)
s = sns.heatmap(df, cmap="Blues",square=True,cbar=False)
s.set_xlabel('Encoder outputs', fontsize=17)
s.set_ylabel('$\longleftarrow$ Decoding step', fontsize=17)

plt.tick_params(axis='both', which='major', labelbottom = True, bottom=False, top = False, labeltop=False);
plt.xticks(rotation=60);

plt.savefig("encoder_cross_attention.png")

layer = 5

cross_attn_contributions = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum')['decoder.encoder_attn'])
cross_attn_contributions = cross_attn_contributions.detach().cpu().numpy()
plt.figure(figsize=(6.2,6.2))
df = pd.DataFrame(cross_attn_contributions[layer],columns=source_sentence + ['Residual'],index=predicted_sentence)
sns.set(font_scale=1.2)
s = sns.heatmap(df, cmap="Blues",square=True,cbar=False)
s.set_xlabel(r'Encoder outputs | Residual $(\tilde{y}^{s}_{t})$', fontsize=15)
s.set_ylabel('$\longleftarrow$ Decoding step', fontsize=17)


plt.tick_params(axis='both', which='major', labelbottom = True, bottom=False, top = False, labeltop=False);
plt.xticks(ticks = plt.gca().get_xticks(), labels=source_sentence + [r'Residual $(\tilde{y}^{s}_{t})$'], rotation=60)
# for i, xtick in enumerate(plt.gca().get_xticklabels()):
#     plt.gca().get_xticklabels()[i].set_color(green_color)
#plt.gca().get_xticklabels()[-1].set_color(red_color)
plt.axvline(x = len(source_sentence)+0.98, lw=1.5, linestyle = '--', color = 'grey')# ymin = 0, ymax = 15
plt.savefig('encoder_plus_residual.png')
print('mean residual',df['Residual'].mean(), 'std', df['Residual'].std())


layer = 5
self_dec_contributions = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum')['decoder.self_attn'])#encoder.self_attn
cross_contributions = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum')['decoder.encoder_attn'])#encoder.self_attn
cross_contributions_layer = cross_contributions[layer]

self_dec_contributions_layer = (self_dec_contributions[layer].transpose(0,1)*cross_contributions_layer[:,-1]).transpose(0,1)
cross_contributions_layer = cross_contributions_layer[:,:-1]
final_cross_contributions = torch.cat((cross_contributions_layer,self_dec_contributions_layer),dim=1)
final_cross_contributions_np = final_cross_contributions.detach().cpu().numpy()
plt.figure(figsize=(15,6),dpi=200)
print(source_sentence, target_sentence)
df = pd.DataFrame(final_cross_contributions_np, columns = source_sentence + target_sentence, index = predicted_sentence)
sns.set(font_scale=1.2)
s = sns.heatmap(df,cmap="Blues",square=True, cbar=False)
s.set_xlabel('Encoder outputs | Decoder layer inputs', fontsize=17)
s.set_ylabel('$\longleftarrow$ Decoding step', fontsize=17)
plt.xticks(rotation=60)
plt.axvline(x = len(source_sentence)+0.98, lw=1.5, linestyle = '--', color = 'grey')# ymin = 0, ymax = 15
src_len = len(source_sentence)
tgt_len = len(predicted_sentence)

plt.savefig('encoder_decoder.png')


relevances_enc_self_attn = hub.get_contribution_rollout(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum')['encoder.self_attn']
layer = -1
# Encoder self-attention relevances in last layer (full encoder ALTI)
plt.figure(figsize=(12,12))
df = pd.DataFrame(relevances_enc_self_attn[layer].cpu().detach().numpy(),columns= source_sentence, index= source_sentence)
sns.set(font_scale=1)
sns.heatmap(df,cmap="Blues",square=True)
plt.gcf().subplots_adjust(bottom=0.2)
plt.savefig('alti_encoder.png')

layer = 4

cross_attn_contributions = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum')['decoder.encoder_attn'])
combined_alti_enc_cross = torch.matmul(cross_attn_contributions[layer][:,:-1],relevances_enc_self_attn[-1])
df = pd.DataFrame(combined_alti_enc_cross.detach().cpu().numpy(), columns=source_sentence, index=predicted_sentence)
sns.set(font_scale=1.2)
sns.heatmap(df,cmap="Blues",square=True)
plt.savefig('alti_encoder_in_decoder.png')

total_alti = hub.get_contribution_rollout(src_tensor, tgt_tensor, 'l1', norm_mode='min_sum')['total']
word_level = True
alignment = False # evaluating alignments, predictions (rows) are the reference, (not showing real interpretations)

alti_result, source_sentence_, predicted_sentence_ = visualize_alti(total_alti, source_sentence,
                                                                    [target_sentence[0]] + ['▁' + target_sentence[1]] + target_sentence[2:], predicted_sentence,
                                                                    word_level, alignment, all_layers = False)
