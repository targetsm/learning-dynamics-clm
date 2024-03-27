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
device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

data_sample = 'interactive' # generate/interactive
teacher_forcing = False # teacher forcing/free decoding

green_color = '#82B366'
red_color = '#B85450'

# Paths
ckpt_dir = "."


hub = FairseqTransformerHub.from_pretrained(
    checkpoint_dir="../tm/checkpoints",
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path="../data-bin/iwslt14.tokenized.de-en/",
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
    i = 120 # index in dataset
    #test_set_dir = europarl_dir / "processed_data/"

    test_set_dir = "./data/iwslt14.tokenized.de-en"
    src = "de"
    tgt = "en"
    tokenizer = "bpe"
    sample = hub.get_interactive_sample(i, test_set_dir, src,
                                        tgt, tokenizer, hallucination=None)
    src_tensor = sample['src_tensor']
    tgt_tensor = sample['tgt_tensor']
    source_sentence = sample['src_tok']
    target_sentence = sample['tgt_tok']

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


layer = 4

cross_attn_contributions = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, 'attn_w', norm_mode='sum_one')['decoder.encoder_attn'])
cross_attn_contributions = cross_attn_contributions.detach().cpu().numpy()
plt.figure(figsize=(20,20))
df = pd.DataFrame(cross_attn_contributions[layer],columns=source_sentence,index=predicted_sentence)
sns.set(font_scale=1.2)
s = sns.heatmap(df, cmap="Blues",square=True,cbar=False)
s.set_xlabel('Encoder outputs', fontsize=17)
s.set_ylabel('$\longleftarrow$ Decoding step', fontsize=17)

plt.tick_params(axis='both', which='major', labelbottom = True, bottom=False, top = False, labeltop=False)
plt.xticks(rotation=60)
plt.savefig('encoder.png')

layer = 4
plt.clf()


cross_attn_contributions = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, 'l1', norm_mode='sum_one')['decoder.encoder_attn'])
cross_attn_contributions = cross_attn_contributions.detach().cpu().numpy()
plt.figure(figsize=(20,20))
print(len(cross_attn_contributions))
print(cross_attn_contributions[layer].shape, len(source_sentence + ['Residual']), len(predicted_sentence + ['']*(512-len(predicted_sentence))))
df = pd.DataFrame(np.transpose(cross_attn_contributions[layer]),columns=source_sentence + ['Residual'], index=predicted_sentence + ['']*(512-len(predicted_sentence)))
sns.set(font_scale=1.2)
s = sns.heatmap(df, cmap="Blues",square=True,cbar=False)
s.set_xlabel(r'Encoder outputs | Residual $(\tilde{y}^{s}_{t})$', fontsize=15)
s.set_ylabel('$\longleftarrow$ Decoding step', fontsize=17)


#plt.tick_params(axis='both', which='major', labelbottom = True, bottom=False, top = False, labeltop=False);
#plt.xticks(ticks = plt.gca().get_xticks(), labels=source_sentence + [r'Residual $(\tilde{y}^{s}_{t})$'], rotation=60)
# for i, xtick in enumerate(plt.gca().get_xticklabels()):
#     plt.gca().get_xticklabels()[i].set_color(green_color)
#plt.gca().get_xticklabels()[-1].set_color(red_color)
#plt.axvline(x = len(source_sentence)+0.98, lw=1.5, linestyle = '--', color = 'grey')# ymin = 0, ymax = 15

plt.savefig('encoder_residual.png')
print('mean residual',df['Residual'].mean(), 'std', df['Residual'].std())



total_alti = hub.get_contribution_rollout(src_tensor, tgt_tensor, 'l1', norm_mode='sum_one')['total']
word_level = True
alignment = False # evaluating alignments, predictions (rows) are the reference, (not showing real interpretations)

alti_result, source_sentence_, predicted_sentence_ = visualize_alti(total_alti, source_sentence, [target_sentence[0]] + ['▁' + target_sentence[1]] + target_sentence[2:], predicted_sentence, word_level, alignment, all_layers = False)

