import operator
from functools import reduce

import torch
#import tensorflow as tf
#from ..ops import record_activations as rec


def jacobian(out, inps):
    """
    :param out: a single tensor functionally dependent on all of inps
    :param inps: list or tuple of tensors w.r.t. which to compute the jacobian
    :returns: a list of same length as inps, where i-th tensor has shape [*out.shape, *inps[i].shape]
    :note: for tf 1.10+ use from tensorflow.python.ops.parallel_for.gradients import jacobian
    """
    flat_out = torch.reshape(out, [-1])
    flat_jac_components = []
    for i in range(flat_out.shape[0]):
        flat_jac_components.append(torch.gradients(flat_out[i], inps))

    jac_components = [torch.reshape(flat_jac, torch.concat([out.shape, inp.shape], axis=0))
                      for flat_jac, inp in zip(flat_jac_components, inps)]
    return jac_components


class LRP:
    """ Helper class for layerwise relevance propagation """
    alpha = 0.5
    beta = 0.5
    eps = 1e-5
    use_alpha_beta = True  # if False, uses simplified LRP rule:  R_i =  R_j * z_ji / ( z_j + eps * sign(z_j) )
    consider_attn_constant = False  # used by MultiHeadAttn, considers gradient w.r.t q/k zeros
    norm_axis = 1

    @classmethod
    def relprop(cls, function, output_relevance, *inps, reference_inputs=None,
                reference_output=None, jacobians=None, batch_axes=(0,), **kwargs):
        """
        computes input relevance given output_relevance using z+ rule
        works for linear layers, convolutions, poolings, etc.
        notation from DOI:10.1371/journal.pone.0130140, Eq 60
        :param function: forward function
        :param output_relevance: relevance w.r.t. layer output
        :param inps: a list of layer inputs
        :param reference_inputs: \hat x, default values used to evaluate bias relevance.
            If specified, must be a tuple/list of tensors of the same shape as inps, default = all zeros.
        :param reference_output: optional pre-computed function(*reference_inputs) to speed up computation
        :param jacobians: optional pre-computed jacobians to speed up computation, same as jacobians(function(*inps), inps)

        """
        assert len(inps) > 0, "please provide at least one input"
        alpha, beta, eps = cls.alpha, cls.beta, cls.eps
        inps = [inp for inp in inps] # ??
        reference_inputs = reference_inputs or tuple(map(torch.zeros_like, inps))
        assert len(reference_inputs) == len(inps)
        output = function(*inps)
        reference_output = reference_output if reference_output is not None else function(*reference_inputs)
        assert isinstance(output, torch.Tensor) and isinstance(reference_output, torch.Tensor)
        flat_output_relevance = torch.reshape(output_relevance, [-1])
        output_size = flat_output_relevance.shape[0]
        
        # 1. compute jacobian w.r.t. all inputs
        jacobians = jacobians if jacobians is not None else jacobian(output, inps)
        # ^-- list of [*output_dims, *input_dims] for each input
        assert len(jacobians) == len(inps)

        jac_flat_components = [torch.reshape(jac, [output_size, -1]) for jac in jacobians]
        # ^-- list of [output_size, input_size] for each input
        flat_jacobian = torch.cat(jac_flat_components, axis=-1)  # [output_size, combined_input_size]
        # 2. multiply jacobian by input to get unnormalized relevances, add bias
        flat_input = torch.cat([torch.reshape(inp, [-1]) for inp in inps], axis=-1)  # [combined_input_size]
        flat_reference_input = torch.cat([torch.reshape(ref, [-1]) for ref in reference_inputs], axis=-1)
        num_samples = reduce(operator.mul, [output.shape[batch_axis] for batch_axis in batch_axes], 1)
        input_size_per_sample = flat_reference_input.shape[0] // num_samples

        flat_bias_impact = torch.reshape(reference_output, [-1]) / float(input_size_per_sample)

        flat_impact = flat_bias_impact[:, None] + flat_jacobian * (flat_input - flat_reference_input)[None, :]
        # ^-- [output_size, combined_input_size], aka z_{j<-i}
        
        if cls.use_alpha_beta:
        # 3. normalize positive and negative relevance separately and add them with coefficients
            flat_positive_impact = torch.clamp(flat_impact, min=0)
            flat_positive_normalizer = torch.sum(flat_positive_impact, dim=cls.norm_axis, keepdim=True) + eps
            flat_positive_relevance = flat_positive_impact / flat_positive_normalizer
            
            flat_negative_impact = torch.clamp(flat_impact, max=0)
            flat_negative_normalizer = torch.sum(flat_negative_impact, dim=cls.norm_axis, keepdim=True) - eps
            flat_negative_relevance = flat_negative_impact / flat_negative_normalizer
            flat_total_relevance_transition = alpha * flat_positive_relevance + beta * flat_negative_relevance
        else:
            flat_impact_normalizer = torch.sum(flat_impact, dim=cls.norm_axis, keepdim=True)
            flat_impact_normalizer += eps * (1. - 2. * float(torch.less(flat_impact_normalizer, 0)))
            flat_total_relevance_transition = flat_impact / flat_impact_normalizer
            # note: we do not use tf.sign(z) * eps because tf.sign(0) = 0, so zeros will not go away

        flat_input_relevance = torch.einsum('o,oi->i', flat_output_relevance, flat_total_relevance_transition)
        # ^-- [combined_input_size]
        
        # 5. unpack flat_inp_relevance back into individual tensors
        input_relevances = []
        offset = 0 # tf.constant(0, dtype=output_size.dtype)
        for inp in inps:
            inp_size = torch.reshape(inp, [-1]).shape[0]
            inp_relevance = torch.reshape(flat_input_relevance[offset: offset + inp_size], inp.shape)
            inp_relevance.view(inp.shape)
            input_relevances.append(inp_relevance)
            offset = offset + inp_size
        #print(input_relevances, input_relevances[0].shape)
        return cls.rescale(output_relevance, *input_relevances, batch_axes=batch_axes, **kwargs)

    @classmethod
    def rescale(cls, reference, *inputs, batch_axes=(0,)):
        assert isinstance(batch_axes, (tuple, list))
        get_summation_axes = lambda tensor: tuple(i for i in range(len(tensor.shape)) if i not in batch_axes)
        ref_scale = torch.sum(abs(reference), dim=get_summation_axes(reference), keepdim=True)
        inp_scales = [torch.sum(abs(inp), dim=get_summation_axes(inp), keepdim=True) for inp in inputs]
        total_inp_scale = sum(inp_scales) + cls.eps
        inputs = [inp * (ref_scale / total_inp_scale) for inp in inputs]
        return inputs[0] if len(inputs) == 1 else inputs


def relprop_add(output_relevance, *inputs, hid_axis=-1, **kwargs):
    """ relprop through elementwise addition of tensors of the same shape """
    #input_shapes = [tf.shape(x) for x in inputs]
    #tiled_input_shape = reduce(tf.broadcast_dynamic_shape, input_shapes)
    #inputs_tiled = [tf.tile(inp, tiled_input_shape // inp_shape) for inp, inp_shape in zip(inputs, input_shapes)]
    #hid_size = tf.shape(inputs[0])[hid_axis]
    #inputs_tiled_flat = [tf.reshape(inp, [-1, hid_size]) for inp in inputs_tiled]
    #output_relevance_flat = tf.reshape(output_relevance, [-1, hid_size])
    #flat_input_relevances_tiled = tf.map_fn(
    #    lambda i: [x[..., 0, :] for x in
    #               LRP.relprop(lambda *inputs: sum(inputs), output_relevance_flat[i, None],
    #                           *[flat_inp[i, None] for flat_inp in inputs_tiled_flat],
    #                           jacobians=[tf.eye(hid_size)[None, :, None, :] for _ in range(len(inputs))],
    #                           batch_axes=(0,), **kwargs)],
    #    elems=tf.range(tf.shape(inputs_tiled_flat[0])[0]),
    #    dtype=[flat_inp.dtype for flat_inp in inputs_tiled_flat],
    #    parallel_iterations=2 ** 10)

    #input_relevances_tiled = list(map(tf.reshape, flat_input_relevances_tiled, [tiled_input_shape] * len(inputs)))
    #input_relevances = list(map(lrp_sum_to_shape, input_relevances_tiled, input_shapes))
    input_relevance = output_relevances
    return input_relevances


def lrp_sum_to_shape(x, new_shape):
    #summation_axes = tf.where(tf.not_equal(new_shape, tf.shape(x)))[..., 0]
    #x_new = tf.reduce_sum(x, axis=summation_axes, keepdims=True)
    #x_new.set_shape([None] * x.shape.ndims)
    x_new = x
    return LRP.rescale(x, x_new, batch_axes=())
