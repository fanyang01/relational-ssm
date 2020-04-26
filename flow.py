from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gnn import GraphNN
from util import trainable_lu_factor, trainable_qr_factor
import util
from bijector import MatvecQR, real_nvp_default_fn

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _make_gate_transform(dim_latent, dim_context, eps=0.001):
    transform = util.mlp_two_layers(
        dim_in=dim_context,
        dim_hid=(2 * dim_context),
        dim_out=(2 * dim_latent),
        act_hid="tanh",
        act_out="sigmoid",
        weight_init='small',
        name="gate_mlp_two_layers"
    )

    def call(local_context):
        gates = tf.math.add(transform(local_context), eps)
        return tf.split(gates, 2, axis=-1)

    return call


def _make_affine_transform(dim_latent, dim_context, i, eps=1E-5):
    transform = util.mlp_two_layers(
        dim_in=dim_context,
        dim_hid=(2 * dim_context),
        dim_out=(2 * dim_latent),
        act_hid="tanh", act_out="linear",
        weight_init='small',
        name="affine_mlp_two_layers_{}".format(i)
    )

    def call(context, base_dist, skip_conn=False):
        scale, shift = tf.split(transform(context), 2, axis=-1)
        scale = tf.math.add(util.exptanh(scale), eps)
        shift = tf.math.tanh(shift)
        if skip_conn:
            shift = tf.math.add(
                shift, tf.math.multiply(
                    tf.math.subtract(1.0, scale),
                    base_dist.mean()
                )
            )
        return scale, shift

    return call


def _make_1x1_affines(dim_latent, dim_context, i):
    to_affine_1 = _make_affine_transform(
        dim_latent, dim_context, "{}_1".format(i)
    )
    to_affine_2 = _make_affine_transform(
        dim_latent, dim_context, "{}_2".format(i)
    )
    return (to_affine_1, to_affine_2)


def _make_1x1_convs(dim_latent, i, factor="qr"):
    if factor == "qr":
        make_fn, bijector_fn = trainable_qr_factor, MatvecQR
    elif factor == "lu":
        make_fn, bijector_fn = trainable_lu_factor, tfp.bijectors.MatvecLU
    else:
        raise ValueError("Unknown factorization: " + factor)

    return (
        bijector_fn(
            *make_fn(event_size=dim_latent, name="conv_1x1_{}_1st".format(i)),
            validate_args=True, name="conv_1x1_{}_1st".format(i)
        ),
        bijector_fn(
            *make_fn(event_size=dim_latent, name="conv_1x1_{}_2nd".format(i)),
            validate_args=True, name="conv_1x1_{}_2nd".format(i)
        ),
    )


def init_real_nvp(num_layers, dim_latent, dim_context, dim_mlp,
                  conv_1x1_factor="qr", name=None):
    assert dim_latent % 2 == 0

    nvp_fn_list, conv_1x1_list, affine_list = [], [], []
    with tf.variable_scope(name, "multi_layer_real_nvp"):
        for i in range(num_layers):
            nvp_fn_list.append(real_nvp_default_fn(
                dim_in=(dim_latent // 2), dim_out=(dim_latent // 2),
                activation=tf.math.tanh,
                name="real_nvp_fn_{}".format(i)
            ))
            conv_1x1_list.append(_make_1x1_convs(
                dim_latent, i, factor=conv_1x1_factor
            ))
            affine_list.append(_make_1x1_affines(dim_latent, dim_context, i))
        to_gate = _make_gate_transform(dim_latent, dim_context)
    return (nvp_fn_list, conv_1x1_list, affine_list, to_gate)


def real_nvp_wrapper(components, context, base_dist, skip_conn=False):
    nvp_fn_list, conv_1x1_list, affine_list, to_gate = components
    assert len(nvp_fn_list) == len(conv_1x1_list)
    num_layers = len(nvp_fn_list)
    assert base_dist.event_shape.ndims == 1
    event_size = base_dist.event_shape.as_list()[-1]
    assert event_size is not None and event_size % 2 == 0
    assert type(base_dist) is tfd.MultivariateNormalDiag or \
        type(base_dist) is tfd.MultivariateNormalDiagPlusLowRank

    bijectors = []
    for i in range(num_layers):
        to_affine_1, to_affine_2 = affine_list[i]
        affine_scale_1, affine_shift_1 = \
            to_affine_1(context, base_dist, skip_conn)
        affine_scale_2, affine_shift_2 = \
            to_affine_2(context, base_dist, skip_conn)

        bijectors.extend([
            conv_1x1_list[i][0],
            tfp.bijectors.Affine(
                scale_diag=affine_scale_1,
                shift=affine_shift_1,
                validate_args=True,
                name="cond_affine_{}_1st".format(i)
            ),
            tfp.bijectors.RealNVP(
                num_masked=(event_size // 2),
                shift_and_log_scale_fn=nvp_fn_list[i],
                validate_args=True,
                name="real_nvp_{}".format(i)
            ),
            conv_1x1_list[i][1],
            tfp.bijectors.Affine(
                scale_diag=affine_scale_2,
                shift=affine_shift_2,
                validate_args=True,
                name="cond_affine_{}_2nd".format(i)
            )
        ])
    if skip_conn:
        gate0, gate1 = to_gate(context)
        bijectors.append(tfp.bijectors.Affine(
            scale_diag=gate0,
            shift=tf.math.multiply(gate1, base_dist.mean()),
            name="skip_conn",
        ))

    # NOTE:
    #   DON'T DO THIS: `tfp.bijectors.Chain(bijectors.reverse())`,
    #   because `L.reverse()` returns `None` :(
    chain = tfp.bijectors.Chain(
        bijectors=list(reversed(bijectors)),
        validate_args=True
    )
    assert chain.forward_min_event_ndims == 1
    assert chain.inverse_min_event_ndims == 1
    return tfd.TransformedDistribution(
        distribution=base_dist, bijector=chain,
        name="transformed_distribution"
    )


def init_perm_equiv_flow(num_layers, dim_latent, dim_context, nvp_gnn_config,
                         conv_1x1_factor="qr", name=None):
    assert dim_latent % 2 == 0
    nvp_gnn_config = nvp_gnn_config.clone()
    nvp_gnn_config.dim_input = (dim_latent // 2) + dim_context
    nvp_gnn_config.dim_global_state = dim_latent
    nvp_gnn_config.layer_norm_in = False
    nvp_gnn_config.layer_norm_out = True
    nvp_gnn_config.skip_conn = True
    nvp_gnn_config.activation = "swish"
    nvp_gnn_config.feed_forward = True
    nvp_gnn_config.feed_forward_act = "tanh"
    nvp_gnn_list, conv_1x1_list, affine_list = [], [], []

    with tf.variable_scope(name, "perm_equiv_flow"):
        for i in range(num_layers):
            nvp_gnn_list.append(GraphNN(
                nvp_gnn_config, dim_out=dim_latent,
                name="nvp_gnn_{}".format(i)
            ))
            conv_1x1_list.append(_make_1x1_convs(
                dim_latent, i, factor=conv_1x1_factor
            ))
            affine_list.append(_make_1x1_affines(dim_latent, dim_context, i))
        to_gate = _make_gate_transform(dim_latent, dim_context)
    return (nvp_gnn_list, conv_1x1_list, affine_list, to_gate)


def perm_equiv_flow_wrapper(components, graph, const_num_nodes,
                            global_context, local_context,
                            base_dist, skip_conn=False):
    nvp_gnn_list, conv_1x1_list, affine_list, to_gate = components
    assert len(nvp_gnn_list) == len(conv_1x1_list)
    num_layers = len(nvp_gnn_list)
    assert type(base_dist) is tfd.Independent and (
        type(base_dist.distribution) is tfd.MultivariateNormalDiag or
        type(base_dist.distribution) is tfd.MultivariateNormalDiagPlusLowRank
    )
    assert base_dist.event_shape.ndims == 2
    assert base_dist.batch_shape.ndims >= 1
    event_size = base_dist.event_shape.as_list()[-1]
    assert event_size is not None and event_size % 2 == 0
    prefix_shape = tf.shape(base_dist.distribution.mean())[:-1]  # (..., B, N)
    flat_event_size = const_num_nodes * event_size

    nodal_half_shape = tf.stack([
        *tf.unstack(prefix_shape), event_size // 2
    ])
    flat_half_shape = tf.stack([
        *tf.unstack(prefix_shape[:-1]), flat_event_size // 2
    ])

    perm = tf.range(flat_event_size)
    perm_2d = tf.reshape(perm, shape=tf.stack([const_num_nodes, event_size]))
    perm_2d_part_1, perm_2d_part_2 = tf.split(perm_2d, 2, axis=-1)
    perm = tf.concat([
        tf.reshape(perm_2d_part_1, [-1]),
        tf.reshape(perm_2d_part_2, [-1])
    ], axis=0)
    # reverse_perm = tf.scatter_nd(
    #     indices=tf.expand_dims(perm, axis=-1),
    #     updates=tf.range(flat_event_size),
    #     shape=[flat_event_size]
    # )
    reverse_perm = tf.math.invert_permutation(perm)

    def make_nvp_fn(gnn_fn):
        def _fn(x, output_units, **condition_kwargs):
            if condition_kwargs:
                raise NotImplementedError("Conditioning not implemented.")
            half_states = tf.reshape(x, shape=nodal_half_shape)
            concat_states = tf.concat([half_states, local_context], axis=-1)
            output = gnn_fn(
                graph=graph, states=concat_states,
                global_states=global_context
            )
            shift, log_scale = tf.split(output, 2, axis=-1)
            return (
                tf.reshape(shift, flat_half_shape),
                tf.reshape(log_scale, flat_half_shape)
            )
        return _fn

    bijectors = []
    for i in range(num_layers):
        to_affine_1, to_affine_2 = affine_list[i]
        affine_scale_1, affine_shift_1 = \
            to_affine_1(local_context, base_dist, skip_conn=False)
        affine_scale_2, affine_shift_2 = \
            to_affine_2(local_context, base_dist, skip_conn=False)

        bijectors.extend([
            conv_1x1_list[i][0],
            tfp.bijectors.Affine(
                scale_diag=affine_scale_1,
                shift=affine_shift_1,
                validate_args=True,
                name="cond_affine_{}_1st".format(i)
            ),
            tfp.bijectors.Reshape(
                event_shape_in=[const_num_nodes, event_size],
                event_shape_out=[flat_event_size],
                validate_args=True,
                name="reshape_{}_in".format(i)
            ),
            tfp.bijectors.Permute(
                perm, axis=-1,
                validate_args=True,
                name="permute_forward_{}".format(i)
            ),
            tfp.bijectors.RealNVP(
                num_masked=(flat_event_size // 2),
                shift_and_log_scale_fn=make_nvp_fn(nvp_gnn_list[i]),
                validate_args=True,
                name="real_nvp_{}".format(i)
            ),
            tfp.bijectors.Permute(
                reverse_perm, axis=-1,
                validate_args=True,
                name="permute_backward_{}".format(i)
            ),
            tfp.bijectors.Reshape(
                event_shape_in=[flat_event_size],
                event_shape_out=[const_num_nodes, event_size],
                validate_args=True,
                name="reshape_{}_out".format(i)
            ),
            conv_1x1_list[i][1],
            tfp.bijectors.Affine(
                scale_diag=affine_scale_2,
                shift=affine_shift_2,
                validate_args=True,
                name="cond_affine_{}_2nd".format(i)
            )
        ])
    if skip_conn:
        gate0, gate1 = to_gate(local_context)
        bijectors.append(tfp.bijectors.Affine(
            scale_diag=gate0,
            shift=tf.math.multiply(gate1, base_dist.mean()),
            name="skip_conn",
        ))

    # NOTE:
    #   DON'T DO THIS: `tfp.bijectors.Chain(bijectors.reverse())`,
    #   because `L.reverse()` returns `None` :(
    chain = tfp.bijectors.Chain(
        bijectors=list(reversed(bijectors)),
        validate_args=True
    )
    assert chain.forward_min_event_ndims == 2
    assert chain.inverse_min_event_ndims == 2

    return tfd.TransformedDistribution(
        distribution=base_dist, bijector=chain,
        name="transformed_distribution"
    )
