import logging
import warnings
from functools import partial
from collections import defaultdict

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple


import numpy as np
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer import TransformerModel
from fairseq import utils

from einops import rearrange

import matplotlib.pyplot as plt
import seaborn as sns

from .lrp import LRP

rc={'font.size': 12, 'axes.labelsize': 10, 'legend.fontsize': 10.0,
    'axes.titlesize': 24, 'xtick.labelsize': 24, 'ytick.labelsize': 24,
    'axes.linewidth': .5, 'figure.figsize': (12,12)}
plt.rcParams.update(**rc)


class FairseqTransformerHub(GeneratorHubInterface):
    ATTN_MODULES = ['encoder.self_attn',
                    'decoder.self_attn',
                    'decoder.encoder_attn']

    def __init__(self, args, task, models):
        super().__init__(args, task, models)
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def from_pretrained(cls, checkpoint_dir, checkpoint_file, data_name_or_path):
        hub_interface = TransformerModel.from_pretrained(checkpoint_dir, checkpoint_file, data_name_or_path)
        return cls(hub_interface.args, hub_interface.task, hub_interface.models)
    
    def encode(self, sentence, dictionary):
        raise NotImplementedError()
    
    def decode(self, tensor, dictionary, as_string=False):
        #tok = dictionary.string(tensor).split()
        tok = []
        for token in torch.squeeze(tensor):
            tok.append(dictionary[token])
        if as_string:
            return ' '.join(tok).replace('▁', ' ')
        else:
            return tok
    
    def get_sample(self, split, index):

        if split not in self.task.datasets.keys():
            self.task.load_dataset(split)

        src_tensor = self.task.dataset(split)[index]['source']
        src_tok = self.decode(src_tensor, self.task.src_dict)
        src_sent = self.decode(src_tensor, self.task.src_dict, as_string=True)

        tgt_tensor = self.task.dataset(split)[index]['target']
        # get_sample returns tensor [..., </s>]
        # we need [</s>, ...] to feed into the decoder
        tgt_tensor = torch.cat([torch.tensor([tgt_tensor[-1]]), tgt_tensor[:-1]])
        tgt_tok = self.decode(tgt_tensor, self.task.tgt_dict)
        tgt_sent = self.decode(tgt_tensor, self.task.tgt_dict, as_string=True)
        

        return {
                'src_tok': src_tok,
                'src_tensor': src_tensor,
                'tgt_tok': tgt_tok,
                'tgt_tensor': tgt_tensor,
                'src_sent': src_sent,
                'tgt_sent': tgt_sent
            }

    def get_interactive_sample(self, i, test_set_dir, src, tgt, tokenizer, hallucination=None):
        """Get interactive sample from tokenized and original word files."""

        test_src_bpe = f'{test_set_dir}/test.{tokenizer}.{src}'
        test_tgt_bpe = f'{test_set_dir}/test.{tokenizer}.{tgt}'
        test_src_word = f'{test_set_dir}/test.{src}'
        test_tgt_word = f'{test_set_dir}/test.{tgt}'

        with open(test_src_bpe, encoding="utf-8") as fbpe:
            # BPE source sentences
            src_bpe_sents = fbpe.readlines()
        with open(test_tgt_bpe, encoding="utf-8") as fbpe:
            # BPE target sentences
            tgt_bpe_sents = fbpe.readlines()
        with open(test_src_word, encoding="utf-8") as fword:
            # Original source sentences
            src_word_sents = fword.readlines()
        with open(test_tgt_word, encoding="utf-8") as fword:
            # Original target sentences
            tgt_word_sents = fword.readlines()

        src_eos_id = self.task.src_dict.eos_index # EOS src token index
        tgt_eos_id = self.task.tgt_dict.eos_index # EOS tgt token index

        src_tok_str = src_bpe_sents[i].strip() # removes leading and trailing whitespaces
        src_tok = src_tok_str.split() + [self.task.src_dict[src_eos_id]]

        # removes leading and trailing whitespaces and add EOS
        tgt_tok_str = tgt_bpe_sents[i].strip()
        tgt_tok = [self.task.tgt_dict[tgt_eos_id]] + tgt_tok_str.split()

        # Add token to beginning of source sentence
        if hallucination is not None:
            src_tok = [hallucination] + ['▁'+ src_tok[0]] + src_tok[1:]
            #tgt_tok = ['<pad>'] + ['▁'+ tgt_tok[0]] + tgt_tok[1:]
        # src_tensor = torch.tensor([self.src_dict.index(t) for t in src_tok] + [eos_id])
        # tgt_tensor = torch.tensor([eos_id] + [self.tgt_dict.index(t) for t in tgt_tok])

        src_tensor = torch.tensor([self.src_dict.index(t) for t in src_tok])
        tgt_tensor = torch.tensor([self.tgt_dict.index(t) for t in tgt_tok])

        if test_src_word and test_tgt_word:
            src_word_sent = src_word_sents[i]
            tgt_word_sent = tgt_word_sents[i]

            return {
                'src_word_sent': src_word_sent,
                'src_tok': src_tok,
                'src_tok_str': src_tok_str,
                'src_tensor': src_tensor,
                'tgt_word_sent': tgt_word_sent,
                'tgt_tok': tgt_tok,
                'tgt_tok_str': tgt_tok_str,
                'tgt_tensor': tgt_tensor
            }

        return {
            'src_word_sent': None,
            'src_tok': src_tok,
            'src_tok_str': src_tok_str,
            'src_tensor': src_tensor,
            'tgt_word_sent': None,
            'tgt_tok': tgt_tok,
            'tgt_tok_str': tgt_tok_str,
            'tgt_tensor': tgt_tensor
        }            
       
    def parse_module_name(self, module_name):
        """ Returns (enc_dec, layer, module)"""
        parsed_module_name = module_name.split('.')
        if not isinstance(parsed_module_name, list):
            parsed_module_name = [parsed_module_name]
            
        if len(parsed_module_name) < 1 or len(parsed_module_name) > 3:
            raise AttributeError(f"'{module_name}' unknown")
            
        if len(parsed_module_name) > 1:
            try:
                parsed_module_name[1] = int(parsed_module_name[1])
            except ValueError:
                parsed_module_name.insert(1, None)
            if len(parsed_module_name) < 3:
                parsed_module_name.append(None)
        else:
            parsed_module_name.extend([None, None])

        return parsed_module_name
    
    def get_module(self, module_name):
        e_d, l, m = self.parse_module_name(module_name)
        module = getattr(self.models[0], e_d)
        if l is not None:
            module = module.layers[l]
            if m is not None:
                module = getattr(module, m)
        else:
            if m is not None:
                raise AttributeError(f"Cannot get'{module_name}'")

        return module

    def trace_forward(self, src_tensor, tgt_tensor):
        r"""Forward-pass through the model.
        Args:
            src_tensor (`tensor`):
                Source sentence tensor.
            tgt_tensor (`tensor`):
                Target sentence tensor (teacher forcing).
        Returns:
            model_output ('tuple'):
                output of the model.
            log_probs:
                log probabilities output by the model.
            encoder_output ('dict'):
                dictionary with 'encoder_out', 'encoder_padding_mask', 'encoder_embedding',
                                'encoder_states', 'src_tokens', 'src_lengths', 'attn_weights'.
            layer_inputs:
                dictionary with the input of the modules of the model.
            layer_outputs:
                dictionary with the input of the modules of the model.
        """
        with torch.no_grad():

            layer_inputs = defaultdict(list)
            layer_outputs = defaultdict(list)

            def save_activation(name, mod, inp, out):
                #print(name, inp, out)
                layer_inputs[name].append(inp)
                layer_outputs[name].append(out)

            handles = {}
             
            for name, layer in self.named_modules():
                handles[name] = layer.register_forward_hook(partial(save_activation, name))
                
            src_tensor = src_tensor.unsqueeze(0).to(self.device)
            tgt_tensor = tgt_tensor.unsqueeze(0).to(self.device)
            
            model_output, encoder_out = self.models[0](src_tensor, src_tensor.size(-1), tgt_tensor, )
            log_probs = self.models[0].get_normalized_probs(model_output, log_probs=True, sample=None)
            
            for k, v in handles.items():
                handles[k].remove()
            self.layer_inputs = layer_inputs
            self.layer_outputs = layer_outputs
            return model_output, log_probs, encoder_out, layer_inputs, layer_outputs

   
    def relprop_residual_new(self, R, name_prev, name_next, main_key):
        
        residual_inp = self.layer_inputs[name_prev[0]][0][0].squeeze(1)
        residual_update = self.layer_outputs[name_next[0]][0].squeeze(1)
        original_scale = torch.sum(abs(R))
        Rinp_residual = 0.0
        R = self.norm_layer.relprop(R)
        Rinp_residual, R = relprop_add(R, residual_inp, residual_update)
        R = self.wrapped_layer.relprop(R)
        if isinstance(R, dict):
            assert main_key is not None
            R_dict = R
            R = R_dict[main_key]        

        pre_residual_scale = torch.sum(abs(R) + abs(Rinp_residual))

        R = R + Rinp_residual
        R = R * pre_residual_scale / torch.sum(torch.abs(R))
        if main_key is not None:
            R_dict = dict(R_dict)
            R_dict[main_key] = R
            total_scale = sum(torch.sum(abs(relevance)) for relevance in R_dict.values())
            R_dict = {key: value * original_scale / total_scale
                      for key, value in R_dict.items()}
            return R_dict
        else:
            return R
    
    def relprop_add(self, output_relevance, name_prev, name_next, hid_axis=-1, **kwargs):
        """ relprop through elementwise addition of tensors of the same shape """
        def lrp_sum_to_shape(x, new_shape):
            summation_axes = torch.where(torch.not_equal(torch.tensor(new_shape),torch.tensor(x.shape)))
            if summation_axes[0].numel():
                summation_axes = summation_axes[:, 0]
                x_new = torch.sum(x, axis=summation_axes, keepdims=True)
                x_new.set_shape([None] * x.shape.ndims)
                return LRP.rescale(x, x_new, batch_axes=())
            else:
                return x
       
        residual_inp = self.layer_inputs[name_prev[0]][0][0].squeeze(1).unsqueeze(0)
        residual_update = self.layer_outputs[name_next[0]][0].squeeze(1).unsqueeze(0)
        inputs = [residual_inp, residual_update]
        input_shapes = [x.shape for x in inputs]
        #tiled_input_shape = reduce(tf.broadcast_dynamic_shape, input_shapes) # no idea what this part does
        tiled_input_shape = torch.broadcast_shapes(*input_shapes)
        
        inputs_tiled = [torch.tile(inp, [tiled_input_shape[s] // input_shapes[0][s] for s in (0,1)]) for inp, inp_shape in zip(inputs, input_shapes)] # this could fail because of tile 
        hid_size = inputs[0].shape[hid_axis]

        inputs_tiled_flat = [torch.reshape(inp, [-1, hid_size]) for inp in inputs_tiled]
        output_relevance_flat = torch.reshape(output_relevance, [-1, hid_size])
        flat_input_relevances_tiled = [[],[]]
        #print(torch.sum(output_relevance_flat))
        for i in range(inputs_tiled_flat[0].shape[0]):
            out = [x[..., 0, :] for x in
                       LRP.relprop(lambda *inputs: sum(inputs), output_relevance_flat[i, None],
                                   *[flat_inp[i, None] for flat_inp in inputs_tiled_flat],
                                   jacobians=None, #[torch.eye(hid_size)[None, :, None, :] for _ in range(len(inputs))],
                                   batch_axes=(0,), **kwargs)]

            flat_input_relevances_tiled[0].append(out[0])
            flat_input_relevances_tiled[1].append(out[1])
        flat_input_relevances_tiled[0] = torch.stack(flat_input_relevances_tiled[0])
        flat_input_relevances_tiled[1] = torch.stack(flat_input_relevances_tiled[1])
        input_relevances_tiled = list(map(torch.reshape, flat_input_relevances_tiled, [tiled_input_shape] * len(inputs)))
        #print(input_relevances_tiled, [torch.sum(x) for x in input_relevances_tiled])
        input_relevances = list(map(lrp_sum_to_shape, input_relevances_tiled, input_shapes))
        
        #input_relevances = [x.shape for x in input_relevances]
        #print('input_relevances', [torch.sum(x) for x in input_relevances], input_relevances)
        return input_relevances

    def relprop_residual(self, R, Rinp_residual, original_scale, name_prev, name_next):
        #original_scale = torch.sum(torch.abs(R)) #TODO Handle scaling for attention
        if original_scale is not None:
            R_dict = R
            R = R_dict['query_inp']

        pre_residual_scale = torch.sum(abs(R)) + torch.sum(abs(Rinp_residual))

        R = R + Rinp_residual
        R = R * pre_residual_scale / torch.sum(torch.abs(R))
        #print('residual_after', R, torch.sum(R))
        if original_scale is not None:
            R_dict = dict(R_dict)
            R_dict['query_inp'] = R
            total_scale = sum(torch.sum(abs(relevance)) for relevance in R_dict.values())
            R_dict = {key: value * original_scale / total_scale
                        for key, value in R_dict.items()}
            return R_dict
        else:
            return R

    def relprop_norm(self, output_relevance, name):
        def _jacobian(inp, self_norm):
            assert len(inp.shape) == 2, "Please reshape your inputs to [batch, dim]"
            batch_size = inp.shape[0]
            hid_size = inp.shape[1]
            centered_inp = (inp - torch.mean(inp, dim=[-1], keepdim=True))
            variance = torch.mean(torch.square(centered_inp), dim=[-1], keepdim=True)
            invstd_factor = torch.rsqrt(variance)
            # note: the code below will compute jacobian without taking self.scale into account until the _last_ line
            jac_out_wrt_invstd_factor = torch.sum(torch.diag(centered_inp), dim=-1, keepdim=True)
            jac_out_wrt_variance = jac_out_wrt_invstd_factor * (-0.5 *(variance + 1e-8) ** (-1.5))[:, :, None,None]
            jac_out_wrt_squared_difference = jac_out_wrt_variance * torch.full([hid_size], 1. / float(hid_size))
            hid_eye = torch.eye(hid_size, hid_size)[None, :, None, :]
            jac_out_wrt_centered_inp = torch.diag(
                invstd_factor) * hid_eye + jac_out_wrt_squared_difference * 2 * centered_inp
            jac_out_wrt_inp = jac_out_wrt_centered_inp - torch.mean(jac_out_wrt_centered_inp, dim=-1,
                                                                    keepdim=True)
            return jac_out_wrt_inp * self_norm.weight[None, :, None, None]
        inp = self.layer_inputs[name[0]][0][0].squeeze(1)
        out = self.layer_outputs[name[0]][0].squeeze(1)

        #print('relprop norm', name[0], 'out', out, 'inp', inp)

        flat_inp = torch.reshape(inp, [-1, inp.shape[-1]]) # shape [inp_size, *dims]
        
        flat_out_relevance = torch.reshape(output_relevance, [-1, output_relevance.shape[-1]])
        self_norm = eval(name[1])
        flat_inp_relevance = []
        for i in range(flat_inp.shape[0]):
            flat_inp_relevance.append(LRP.relprop(self_norm, flat_out_relevance[i, None], 
                flat_inp[i, None], jacobians=None,#_jacobian(flat_inp[i, None], self_norm),
                batch_axes=(0,))[0])
        flat_inp_relevance = torch.stack(flat_inp_relevance).unsqueeze(0)
        #print(flat_inp_relevance.shape, output_relevance.shape)
        #print('norm_before', torch.sum(flat_inp_relevance), torch.sum(output_relevance))
        input_relevance = LRP.rescale(output_relevance, flat_inp_relevance)
        #print('norm_after', torch.sum(input_relevance))
        return input_relevance

    def relprop_ffn(self, output_relevance, name):
        """
        computes input relevance given output_relevance
        :param output_relevance: relevance w.r.t. layer output, [*dims, out_size]
        notation from DOI:10.1371/journal.pone.0130140, Eq 60
        """
        inp = self.layer_inputs[name[0]][0][0].squeeze(1)
        out = self.layer_outputs[name[0]][0].squeeze(1)
        #print(self.layer_inputs[name[0]], name[0])
        #print('relprop ffn',   name[0], 'out', out, 'inp', inp)
        # inp: [*dims, inp_size], out: [*dims, out_size]

        linear_self = eval(name[1])

        # note: we apply relprop for each independent sample in order to avoid quadratic memory requirements
        flat_inp = torch.reshape(inp, [-1, inp.shape[-1]]) # shape [inp_size, *dims]

        flat_out_relevance = torch.reshape(output_relevance, [-1, output_relevance.shape[-1]])

        
        flat_inp_relevance = []
        for i in range(flat_inp.shape[0]):
            flat_inp_relevance.append(LRP.relprop(linear_self, flat_out_relevance[i, None], flat_inp[i, None],
                                      jacobians=[linear_self.weight.data.T[None, :, None, :]], batch_axes=(0,))[0])
        flat_inp_relevance = torch.stack(flat_inp_relevance).unsqueeze(0)
        input_relevance = LRP.rescale(output_relevance, flat_inp_relevance)
        return input_relevance

    def __get_attn_weights_module(self, layer_outputs, module_name):
        enc_dec_, l, attn_module_ = self.parse_module_name(module_name)
        
        attn_module = self.get_module(module_name)
        num_heads = attn_module.num_heads
        head_dim = attn_module.head_dim
        k = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.k_proj"][0]
        q = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0]

        q, k = map(
            lambda x: rearrange(
                x,
                't b (n_h h_d) -> (b n_h) t h_d',
                n_h=num_heads,
                h_d=head_dim
            ),
            (q, k)
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if enc_dec_ == 'decoder' and attn_module_ == 'self_attn':
            tri_mask = torch.triu(torch.ones_like(attn_weights), 1).bool()
            attn_weights[tri_mask] = -1e9

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = rearrange(
            attn_weights,
            '(b n_h) t_q t_k -> b n_h t_q t_k',
            n_h=num_heads
        )
        return attn_weights
    
    def attn_core(self, layer_inputs, layer_outputs, contrib_type, module_name):
        # Get info about module: encoder, decoder, self_attn, cross-attn
        #enc_dec_, l, attn_module_ = self.parse_module_name(module_name)
        
        # Get info about LN (Pre-LN or Post-LN)
        if enc_dec_ == 'encoder':
            pre_layer_norm = self.args.encoder_normalize_before
        else:
            pre_layer_norm = self.args.decoder_normalize_before
        
        attn_w = self.__get_attn_weights_module(layer_outputs, ) # (batch_size, num_heads, src:len, src_len)
        #print('attn_w', attn_w.shape)
        def l_transform(x, w_ln):
            '''Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation.'''
            ln_param_transf = torch.diag(w_ln)
            ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - \
                1 / w_ln.size(0) * torch.ones_like(ln_param_transf).to(w_ln.device)

            out = torch.einsum(
                '... e , e f , f g -> ... g',
                x,
                ln_mean_transf,
                ln_param_transf
            )
            return out

        attn_module = self.get_module(module_name)
        w_o = attn_module.out_proj.weight
        b_o = attn_module.out_proj.bias
        
        ln = self.get_module(f'{module_name}_layer_norm')
        w_ln = ln.weight.data
        b_ln = ln.bias
        eps_ln = ln.eps

        ## LN2
        ln2 = self.get_module(f'{enc_dec_}.{l}.final_layer_norm')
        w_ln2 = ln.weight.data
        b_ln2 = ln.bias
        eps_ln2 = ln.eps
        
        in_q = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0][0].transpose(0, 1)
        in_v = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.v_proj"][0][0].transpose(0, 1)
        in_res = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0].transpose(0, 1)
        ##
        in_res2 = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)
        #print(in_v.shape, in_q.shape, in_res.shape, in_res2.shape)

        if "self_attn" in attn_module_:
            if pre_layer_norm:
                residual_ = torch.einsum('sk,bsd->bskd', torch.eye(in_res.size(1)).to(in_res.device), in_res)
            else:
                residual_ = torch.einsum('sk,bsd->bskd', torch.eye(in_q.size(1)).to(in_res.device), in_q)
        else:
            if pre_layer_norm:
                residual_ = in_res
            else:
                residual_ = in_q
             
        v = attn_module.v_proj(in_v)
        #print('v', v.shape)
        v = rearrange(
            v,
            'b t_v (n_h h_d) -> b n_h t_v h_d',
            n_h=attn_module.num_heads,
            h_d=attn_module.head_dim
        )
        #print('v_rearr', v.shape)
        w_o = rearrange(
            w_o,
            'out_d (n_h h_d) -> n_h h_d out_d',
            n_h=attn_module.num_heads,
        )
        #print('w_o_rearr', w_o.shape)
        
        attn_v_wo = torch.einsum(
            'b h q k , b h k e , h e f -> b q k f',
            attn_w,
            v,
            w_o
        )

        # Add residual
        if "self_attn" in attn_module_:
            out_qv_pre_ln = attn_v_wo + residual_
        # Concatenate residual in cross-attention (as another value vector)
        else:
            #print('attn_v_wo', attn_v_wo.shape, residual_.shape)
            out_qv_pre_ln = torch.cat((attn_v_wo,residual_.unsqueeze(-2)),dim=2)
        
        # Assert MHA output + residual is equal to 1st layer normalization input
        out_q_pre_ln = out_qv_pre_ln.sum(-2) +  b_o

        #### NEW
        if pre_layer_norm==False:
            # In post-ln we compare with the input of the first layernorm
            out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0].transpose(0, 1)
        else:
            if 'encoder' in enc_dec_:
                # Encoder (self-attention) -> final_layer_norm
                out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)
            else:
                if "self_attn" in attn_module_:
                    # Self-attention decoder -> encoder_attn_layer_norm
                    out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.encoder_attn_layer_norm"][0][0].transpose(0, 1)
                else:
                    # Cross-attention decoder -> final_layer_norm
                    out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)
        #### NEW

        # if pre_layer_norm:
        #     if 'encoder' in enc_dec_:
        #          # Encoder (self-attention) -> final_layer_norm
        #         out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)
                
        #     else:
        #         if "self_attn" in attn_module_:
        #             # Self-attention decoder -> encoder_attn_layer_norm
        #             out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.encoder_attn_layer_norm"][0][0].transpose(0, 1)
        #         else:
        #             # Cross-attention decoder -> final_layer_norm
        #             out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)
                
        # else:
        #     # In post-ln we compare with the input of the first layernorm
        #     out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0].transpose(0, 1)
        
        assert torch.dist(out_q_pre_ln_th, out_q_pre_ln).item() < 1e-3 * out_q_pre_ln.numel()
        
        if pre_layer_norm:
            transformed_vectors = out_qv_pre_ln
            resultant = out_q_pre_ln
        else:
            ln_std_coef = 1/(out_q_pre_ln_th + eps_ln).std(-1).view(1,-1, 1).unsqueeze(-1) # (batch,src_len,1,1)
            transformed_vectors = l_transform(out_qv_pre_ln, w_ln)*ln_std_coef # (batch,src_len,tgt_len,embed_dim)
            dense_bias_term = l_transform(b_o, w_ln)*ln_std_coef # (batch,src_len,1,embed_dim)
            attn_output = transformed_vectors.sum(dim=2) # (batch,seq_len,embed_dim)
            resultant = attn_output + dense_bias_term.squeeze(2) + b_ln # (batch,seq_len,embed_dim)
            
            # Assert resultant (decomposed attention block output) is equal to the real attention block output
            out_q_th_2 = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0].transpose(0, 1)
            assert torch.dist(out_q_th_2, resultant).item() < 1e-3 * resultant.numel()
            #print('res:', resultant.shape, 'trans:', transformed_vectors.shape)

        if contrib_type == 'l1':
            #print(transformed_vectors.shape)
            #print(resultant.shape)
            #print(torch.norm(transformed_vectors.sub(resultant.unsqueeze(2)), p=1, dim=-1).shape)
            contributions = -torch.norm(transformed_vectors.sub(resultant.unsqueeze(2)), p=1, dim=-1)
            #print(contributions.shape)
            #contributions = -F.pairwise_distance(transformed_vectors.transform, resultant.unsqueeze(2), p=1)
            #print('final_contrib', contributions.shape)
            resultants_norm = torch.norm(torch.squeeze(resultant),p=1,dim=-1)
            #print('final_res', resultants_norm.shape)

        elif contrib_type == 'l2':
            contributions = -F.pairwise_distance(transformed_vectors, resultant.unsqueeze(2), p=2)
            resultants_norm = torch.norm(torch.squeeze(resultant),p=2,dim=-1)
            #resultants_norm=None
        elif contrib_type == 'koba':
            contributions = torch.norm(transformed_vectors, p=2, dim=-1)
            return contributions, None
        else:
            raise ArgumentError(f"contribution_type '{contrib_type}' unknown")
    
        return contributions, resultants_norm

    def relprop_attn(self, R, name, is_combined=False):
        def _split_heads(x, num_heads):
            """
            Split channels (dimension 3) into multiple heads (dimension 1)
            input: (batch_size * ninp * inp_dim)
            output: (batch_size * n_heads * ninp * (inp_dim/n_heads))
            """
            old_shape = [int(i) for i in x.shape]
            dim_size = old_shape[-1]
            new_shape = old_shape[:-1] + [num_heads] + [dim_size // num_heads if dim_size else None]
            ret = torch.reshape(x, old_shape[:-1] + [num_heads, old_shape[-1] // num_heads])
            ret = torch.reshape(ret, new_shape)
            return torch.transpose(ret, 1, 2)  # [batch_size * n_heads * ninp * (hid_dim//n_heads)]
        def _combine_heads(x):
            """
            Inverse of split heads
            input: (batch_size * n_heads * ninp * (inp_dim/n_heads))
            out: (batch_size * ninp * inp_dim)
            """
            x = torch.transpose(x, 2, 1)
            old_shape = x.shape
            a, b = old_shape[-2:]
            new_shape = list(old_shape[:-2]) + [a * b if a and b else None]
            ret = torch.reshape(x, list(x.shape[:-2]) + [x.shape[-2] * x.shape[-1]])
            ret = torch.reshape(x, new_shape)
            return ret
        enc_dec_ = name[0].split('.')[2]
        l = name[0].split('.')[4]
        attn_module_ = name[0].split('.')[5]

        exit()
        attn_self = eval(name[1])
        #print(attn_self.forward())
 
        original_scale = torch.sum(abs(R))
        #print(R.shape)
        R = self.relprop_ffn(R, [x + '.out_proj' for x in name])
        R_split = _split_heads(R, attn_self.num_heads)
        print('original',self.layer_inputs[name[0]+'.out_proj'][0][0], self.layer_inputs[name[0]+'.out_proj'][0][0].shape)
        # note: we apply relprop for each independent sample and head in order to avoid quadratic memory growth
        q = self.layer_outputs[name[0]+'.q_proj'][0]#.transpose(0,1)
        k = self.layer_outputs[name[0]+'.k_proj'][0]#.transpose(0,1)
        v = self.layer_outputs[name[0]+'.v_proj'][0]#.transpose(0,1)
        #print(self.attn_core(attn_self, q, k, v)[0], self.attn_core(attn_self, q, k, v)[0].shape)
        #print('attention',  name[0], 'q', q.shape,'k', k.shape,'v', v.shape)
        q, k, v = map(
            lambda x: rearrange(
                x,
                't b (n_h h_d) -> (b n_h) t h_d',
                n_h=1, #attn_self.num_heads,
                h_d=attn_self.head_dim*attn_self.num_heads,
            ),
            (q, k, v)
        ) 

        attn_weights = torch.bmm(q, k.transpose(1, 2))


        #print('attention',  name[0], 'q', q.shape,'k', k.shape,'v', v.shape)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if  'decoder' in name[0] and 'self_attn' in name[0]:
            flat_attn_mask = torch.tril(torch.ones_like(attn_weights), 0)
            #print('flat_attn_mask', flat_attn_mask)
        else:
            flat_attn_mask = torch.zeros_like(attn_weights)
        
        q_flat, k_flat, v_flat = q, k, v
        dim_per_head = v_flat.shape[-1]
        batch_size, n_heads, n_q = R_split.shape[0], R_split.shape[1], R_split.shape[2]
        n_kv = flat_attn_mask.shape[2]
        R_flat = torch.reshape(R_split, [-1, n_q, dim_per_head])
        # ^-- *_flat variables are of shape: [(batch * n_heads), n_q, dim per head]
        
        #attn_jacobian = self._attn_head_jacobian_simple if LRP.consider_attn_constant else self._attn_head_jacobian
        attn_jacobian = None
        flat_relevances = []
        #flat_attn_mask.requires_grad_()
        #print(flat_attn_mask.shape, q.shape
        flat_relevances = LRP.relprop(
                lambda q, k, v: self.attn_core(attn_self, q.transpose(0,1), k.transpose(0,1), v.transpose(0,1), attn_mask=flat_attn_mask.squeeze(0)),
                R, q, k, v,
                jacobians=None, #attn_jacobian(q_flat[i, None], k_flat[i, None], v_flat[i, None], flat_attn_mask[i, None]),
                batch_axes=(0,))
        
        #for i in range(q_flat.shape[0]):
        #    flat_relevances.append(LRP.relprop(
        #        lambda q, k, v: self.attn_core(attn_self, q, k, v, attn_mask=flat_attn_mask[i, None]),
        #        R_flat[i, None], q_flat[i, None], k_flat[i, None], v_flat[i, None],
        #        jacobians=None, #attn_jacobian(q_flat[i, None], k_flat[i, None], v_flat[i, None], flat_attn_mask[i, None]),
        #        batch_axes=(0,)))
        #print([torch.sum(x, -1) for x in flat_relevances])
        #print([x.shape for x in flat_relevances])
        #flat_relevances = list(map(list, zip(*flat_relevances)))
        #flat_relevances = [torch.stack(rel) for rel in flat_relevances]
        #print([x.shape for x in flat_relevances])
        #print([[torch.sum(y, -1) for y in x] for x in flat_relevances])
        #exit()
        #Rq, Rk, Rv = [_combine_heads(torch.reshape(rel_flat, [batch_size, n_heads, -1, dim_per_head]))
                      #for rel_flat in flat_relevances]
        Rq, Rk, Rv = flat_relevances[0], flat_relevances[1], flat_relevances[2]
        Rq, Rk, Rv = LRP.rescale(R, Rq, Rk, Rv, batch_axes=(0,))
        #print(Rq.shape, Rk.shape, Rv.shape)
        if is_combined:
            Rq = self.relprop_ffn(Rq, [x + '.q_proj' for x in name])
            Rk = self.relprop_ffn(Rk, [x + '.k_proj' for x in name])
            Rv = self.relprop_ffn(Rv, [x + '.v_proj' for x in name])
            #print('Rq', Rq, 'Rk', Rk, 'Rv', Rv)

            attn_scale = torch.sum(abs(Rq) + abs(Rk) + abs(Rv))
            Rinp = Rq + Rk + Rv
            Rinp = Rinp * original_scale / attn_scale
            return Rinp
        else:
            Rq = self.relprop_ffn(Rq, [x + '.q_proj' for x in name])
            Rk = self.relprop_ffn(Rk, [x + '.k_proj' for x in name])
            Rv = self.relprop_ffn(Rv, [x + '.v_proj' for x in name])
            #print('Rq', Rq, 'Rk', Rk, 'Rv', Rv)
            attn_scale = torch.sum(torch.abs(Rq)) + torch.sum(torch.abs(Rk)) + torch.sum(torch.abs(Rv))
            Rqinp = Rq * original_scale / attn_scale
            Rkvinp =  (Rk + Rv) * original_scale / attn_scale
            return {'query_inp': Rqinp, 'kv_inp': Rkvinp}
    
    def get_name(self, i, string, enc_dec):
        return (f'models.0.{enc_dec}.layers.{i}.{string}', f'self.models[0].{enc_dec}.layers[{i}].{string}')

    def relprop_decode(self, R):
        """ propagates relevances from rdo to output embeddings and encoder state """
        #if self.normalize_out:
        #    R = self.dec_out_norm.relprop(R)

        R_enc = 0.0
        R_enc_scale = 0.0
        R_orig = R
        for i in range(len(self.models[0].decoder.layers))[::-1]:
            print(i, torch.sum(R), R)
            R = self.relprop_norm(R, self.get_name(i,'final_layer_norm',  'decoder'))
            print('final_norm', torch.sum(R), R)
            R, R_res = self.relprop_add(R, self.get_name(i,'fc1', 'decoder'), self.get_name(i,'fc2', 'decoder'))
            #exit()
            print('add', torch.sum(R), torch.sum(R_res), R)
            R = self.relprop_ffn(R, self.get_name(i,'fc2', 'decoder'))
            R = self.relprop_ffn(R, self.get_name(i,'fc1', 'decoder'))
            print('ffn', torch.sum(R), R)
            R = self.relprop_residual(R, R_res, None, self.get_name(i,'fc1', 'decoder'), self.get_name(i,'fc2', 'decoder'))
            print('residual', torch.sum(R), R)
            R = self.relprop_norm(R, self.get_name(i,'encoder_attn_layer_norm', 'decoder'))
            print('norm', torch.sum(R), R)
            #exit()
            original_scale = torch.sum(R)
            R, R_res = self.relprop_add(R, self.get_name(i,'encoder_attn.q_proj', 'decoder'), self.get_name(i,'encoder_attn.out_proj', 'decoder'))
            print('add', torch.sum(R), torch.sum(R_res), R, R_res)
            #exit()
            #possibly another layernorm here attn_ln 
            relevance_dict = self.relprop_attn(R, self.get_name(i,'encoder_attn', 'decoder'))
            print('encoder_attn', torch.sum(relevance_dict['query_inp']), torch.sum(relevance_dict['kv_inp']), relevance_dict)
            exit()
            relevance_dict = self.relprop_residual(relevance_dict, R_res, original_scale, self.get_name(i,'encoder_attn', 'decoder'), self.get_name(i,'encoder_attn', 'decoder'))
            print('residual', torch.sum(relevance_dict['query_inp']), torch.sum(relevance_dict['kv_inp']), relevance_dict)
            R = relevance_dict['query_inp']
            R_enc += relevance_dict['kv_inp']
            R_enc_scale += torch.sum(torch.abs(relevance_dict['kv_inp']))
            print('residual_added', 'R', torch.sum(R), 'R_enc', torch.sum(R_enc), relevance_dict)
            #exit()
            R = self.relprop_norm(R, self.get_name(i,'self_attn_layer_norm', 'decoder'))
            print('norm', torch.sum(R), R)
            R, R_res = self.relprop_add(R, self.get_name(i,'self_attn.q_proj', 'decoder'), self.get_name(i,'self_attn.out_proj', 'decoder'))
            print('add', torch.sum(R), torch.sum(R_res), R, R_res)
            R = self.relprop_attn(R, self.get_name(i,'self_attn', 'decoder'), True)
            print('self_attn', torch.sum(R), R)
            R = self.relprop_residual(R, R_res, None, self.get_name(i,'self_attn', 'decoder'), self.get_name(i,'self_attn', 'decoder'))
            print('residual', torch.sum(R), R)
        # shift left: compensate for right shift
        R_crop = F.pad(input=R, pad=(0, 0, 0, 1, 0, 0), mode='constant', value=0)[:, 1:, :]
        return {'emb_out': R_crop, 'enc_out': R_enc * R_enc_scale / torch.sum(torch.abs(R_enc)),
                'emb_out_before_crop': R}
        

    def relprop_encode(self, R):
        """ propagates relevances from enc_out to emb_inp """
        for i in range(len(self.models[0].encoder.layers))[::-1]:
            print('encoder', i, torch.sum(R, -1))
            R = self.relprop_norm(R, self.get_name(i,'final_layer_norm', 'encoder'))
            print('norm', torch.sum(R), R)
            R, R_res = self.relprop_add(R, self.get_name(i,'fc1', 'encoder'), self.get_name(i,'fc2', 'encoder'))
            print('add', torch.sum(R), torch.sum(R_res), R, R_res)
            R = self.relprop_ffn(R, self.get_name(i,'fc2', 'encoder'))
            print('fc2', torch.sum(R), R)
            R = self.relprop_ffn(R, self.get_name(i,'fc1', 'encoder'))
            print('fc1', torch.sum(R), R)
            R = self.relprop_residual(R, R_res, None, self.get_name(i,'fc1', 'encoder'), self.get_name(i,'fc2', 'encoder'))
            print('residual', torch.sum(R), R)
            R = self.relprop_norm(R, self.get_name(i,'self_attn_layer_norm', 'encoder'))
            print('norm', torch.sum(R), R)
            R, R_res = self.relprop_add(R, self.get_name(i,'self_attn.q_proj', 'encoder'), self.get_name(i,'self_attn.out_proj', 'encoder'))
            print('add', torch.sum(R), torch.sum(R_res), R, R_res)
            R = self.relprop_attn(R, self.get_name(i,'self_attn', 'encoder'), True)
            print('attn', torch.sum(R), R)
            R = self.relprop_residual(R, R_res, None, self.get_name(i,'self_attn', 'encoder'), self.get_name(i,'self_attn', 'encoder'))
            print('residual', torch.sum(R), R)
        return R

