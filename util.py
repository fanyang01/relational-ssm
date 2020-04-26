from __future__ import absolute_import
from __future__ import print_function

import functools
import itertools

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.python.util import nest

tfd = tfp.distributions

LSTM = "LSTM"
GRU = "GRU"

SOFTMAX_MASK_MULTIPLIER = -10000.0

initializer = keras.initializers.glorot_normal

HID_ACTIVATION = "tanh"


class PartialLocScaleDist(object):
    def __init__(self, loc, fn, batch_ndims=0):
        self._loc = loc
        self._fn = fn
        self._batch_ndims = batch_ndims

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = value

    def build(self):
        dist = self._fn(loc=self._loc)
        if self._batch_ndims <= 0:
            return dist
        return tfd.Independent(
            dist, reinterpreted_batch_ndims=self._batch_ndims
        )


def dim(x):
    d = x.shape.as_list()[-1]
    assert d is not None
    return d


def float(x):
    return tf.cast(x, tf.float32)


def append_dims(x, count=1):
    for i in range(count):
        x = tf.expand_dims(x, axis=-1)
    return x


def broadcast_concat(batched_x, y):
    assert batched_x.shape.ndims >= y.shape.ndims
    broadcast_y = tf.math.add(y, tf.expand_dims(
        tf.zeros(tf.shape(batched_x)[:-1]), axis=-1
    ))
    return tf.concat([batched_x, broadcast_y], axis=-1)


def count_rnn_states(rnn_cell):
    return len(nest.flatten(rnn_cell.state_size))


def concat_rnn_states(states):
    return tf.concat(nest.flatten(states), axis=-1)


def pack_rnn_states(rnn_cell, flat_states):
    state_size = rnn_cell.state_size
    num_split = count_rnn_states(rnn_cell)
    states = tf.split(flat_states, num_split, axis=-1)
    return nest.pack_sequence_as(state_size, states)


def extract_rnn_output(concat_states, num_layers, cell_type):
    if cell_type is not None:
        cell_type = cell_type.upper()
    if cell_type != LSTM and cell_type != GRU:
        return concat_states
    num_state_per_layer = (2 if cell_type == LSTM else 1)
    num_split = num_layers * num_state_per_layer
    rnn_states = tf.split(concat_states, num_split, axis=-1)
    return rnn_states[-num_state_per_layer]


def left_shift_and_pad(sequence, element=None):
    if element is None:
        element = tf.zeros_like(sequence[0])
    element = tf.expand_dims(element, axis=0)
    return tf.concat([sequence[1:], element], axis=0)


def print_summary(prefix, inputs):
    flatten = tf.reshape(inputs, [-1])
    mean, variance = tf.nn.moments(flatten, axes=[0])
    p10 = tfd.percentile(flatten, q=10.0, axis=-1)
    median = tfd.percentile(flatten, q=50.0, axis=-1)
    p90 = tfd.percentile(flatten, q=90.0, axis=-1)
    maximum, _ = tf.math.top_k(tf.math.abs(flatten))
    print_op = tf.print(
        prefix + " :[mean var p10 p50 p90 max] =",
        mean, variance, p10, median, p90, maximum[0]
    )
    with tf.control_dependencies([print_op]):
        return tf.identity(inputs)


def select(cond, x, y, expand=False, set_shape=False):
    mask = tf.cast(cond, tf.float32)
    if expand:
        assert x.shape.ndims == y.shape.ndims
        while mask.shape.ndims < x.shape.ndims:
            mask = tf.expand_dims(mask, axis=-1)
    result = tf.math.add(
        tf.math.multiply(mask, x),
        tf.math.multiply(1.0 - mask, y)
    )
    if set_shape:
        result.set_shape(y.shape)
    return result


def select_nested(cond, x, y, expand=False, set_shape=False):
    if not (type(x) is list or type(x) is tuple):
        return select(cond, x, y, expand=expand, set_shape=set_shape)
    result = []
    for elem_x, elem_y in zip(x, y):
        result.append(select_nested(
            cond, elem_x, elem_y,
            expand=expand, set_shape=set_shape
        ))
    return tuple(result)


def nested_set_shape_like(x, y):
    if not (type(x) is list or type(x) is tuple):
        x.set_shape(y.shape)
        return x
    result = []
    for elem_x, elem_y in zip(x, y):
        result.append(nested_set_shape_like(elem_x, elem_y))
    return tuple(result)


def exptanh(x, alpha=2.5):
    return tf.math.exp(alpha * tf.math.tanh(x / (2.0 * alpha)))


def lookup_activation_fn(name):
    if name == "linear":
        return tf.identity
    elif name == "tanh":
        return tf.math.tanh
    elif name == "relu":
        return tf.nn.relu
    elif name == "leaky_relu":
        return tf.nn.leaky_relu
    elif name == "swish":
        return tf.nn.swish
    elif name == "softplus":
        return tf.math.softplus
    elif name == "zero":
        return tf.zeros_like
    elif name == "square":
        return tf.math.square
    else:
        raise ValueError("unknown activations")


def make_activation_layer(name, input_shape=None):
    kwargs = {} if input_shape is None else dict(input_shape=input_shape)
    if name == "leaky_relu":
        return keras.layers.LeakyReLU(0.2, **kwargs)
    elif name == "square":
        return keras.layers.Lambda(lambda x: tf.math.square(x))
    elif name == "swish":
        return keras.layers.Lambda(lambda x: tf.nn.swish(x))
    return keras.layers.Activation(name, **kwargs)


def make_lstm_cells(num_layers, dim_in, cell_size, name="lstm_cells"):
    with tf.variable_scope(name):
        if num_layers == 1:
            return keras.layers.LSTMCell(
                cell_size, input_shape=(dim_in,),
                recurrent_activation="sigmoid",
                name="lstm_cell"
            )
        input_cell = keras.layers.LSTMCell(
            cell_size, input_shape=(dim_in,),
            recurrent_activation="sigmoid",
            name="lstm_cell_0"
        )
        hidden_cells = [
            keras.layers.LSTMCell(
                cell_size, input_shape=(cell_size,),
                recurrent_activation="sigmoid",
                name="lstm_cell_{}".format(i + 1)
            )
            for i in range(num_layers - 1)
        ]
        cells = [input_cell, *hidden_cells]
        return keras.layers.StackedRNNCells(cells)


def make_trainable_gmm(dim, num_components, name="GMM"):
    with tf.variable_scope(name):
        loc = tf.get_variable(
            name="loc", shape=[num_components, dim],
            trainable=True, initializer=initializer()
        )
        raw_scale_diag = tf.get_variable(
            name="raw_scale_diag", shape=[num_components, dim],
            trainable=True, initializer=initializer()
        )
        mixture_logits = tf.get_variable(
            name="mixture_logits", shape=[num_components],
            trainable=True, initializer=initializer()
        )

    return tfd.MixtureSameFamily(
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=tf.nn.softplus(raw_scale_diag)
        ),
        mixture_distribution=tfd.Categorical(logits=mixture_logits),
        name=name
    )


def broadcast_gmm(gmm_dist, batch_shape):
    assert type(gmm_dist) is tfd.MixtureSameFamily and \
        type(gmm_dist.mixture_distribution) is tfd.Categorical and \
        type(gmm_dist.components_distribution) is tfd.MultivariateNormalDiag
    mvns = gmm_dist.components_distribution
    broadcast_add_oprand = tf.zeros(tf.stack([
        *tf.unstack(batch_shape),
        *(mvns.batch_shape.as_list()),
        *(mvns.event_shape.as_list())
    ]))
    new_locs = tf.math.add(broadcast_add_oprand, mvns.mean())
    new_scales = tf.math.add(broadcast_add_oprand, mvns.stddev())

    cat = gmm_dist.mixture_distribution
    broadcast_add_oprand = tf.zeros(tf.stack([
        *tf.unstack(batch_shape),
        *(cat.batch_shape.as_list()),
        cat.event_size
    ]))
    new_logits = tf.math.add(broadcast_add_oprand, cat.logits)

    return tfd.MixtureSameFamily(
        components_distribution=tfd.MultivariateNormalDiag(
            loc=new_locs, scale_diag=tf.nn.softplus(new_scales)
        ),
        mixture_distribution=tfd.Categorical(logits=new_logits),
        name=gmm_dist.name
    )


def mlp_one_layer(dim_in, dim_out, act_out="linear", name="mlp_one_layer"):
    return keras.Sequential([
        keras.layers.Dense(dim_out, input_dim=dim_in),
        make_activation_layer(act_out)
    ], name=name)


def mlp_two_layers(dim_in, dim_hid, dim_out,
                   act_hid=HID_ACTIVATION, act_out="linear",
                   weight_init='glorot_uniform',
                   name="mlp_two_layers"):
    if weight_init == 'small':
        weight_init = keras.initializers.truncated_normal(stddev=0.0025)
    return keras.Sequential([
        keras.layers.Dense(dim_hid, input_dim=dim_in),
        make_activation_layer(act_hid),
        keras.layers.Dense(dim_out, kernel_initializer=weight_init),
        make_activation_layer(act_out)
    ], name=name)


def zero_activation(dim_o):
    def zero(input):
        return tf.zeros(tf.stack([
            *tf.unstack(tf.shape(input)[:-1]), dim_o
        ]))
    return zero


def _new_layer(dim_i, dim_o, act_i, act_o, name):
    if act_o == "zero":
        return zero_activation(dim_o)
    return keras.Sequential([
        make_activation_layer(act_i, input_shape=(dim_i,)),
        keras.layers.Dense(dim_o),
        make_activation_layer(act_o)
    ], name=name)


def _make_cond_two_param_dist(mlp, dim_from, dim_to, make_dist,
                              act_0="linear", act_1="linear", layer_norm=False,
                              name="cond_two_param_dist"):
    with tf.variable_scope(name):
        layer_norm_fn = tf.identity if not layer_norm \
            else layer_norm_1d(dim_from, trainable=True)
        to_param_0 = _new_layer(
            dim_i=dim_from, dim_o=dim_to, act_i=HID_ACTIVATION, act_o=act_0,
            name="to_param_0"
        )
        to_param_1 = _new_layer(
            dim_i=dim_from, dim_o=dim_to, act_i=HID_ACTIVATION, act_o=act_1,
            name="to_param_1"
        )

    def cond_dist(input):
        hidden = layer_norm_fn(mlp(input))
        param_0, param_1 = to_param_0(hidden), to_param_1(hidden)
        return make_dist(param_0, param_1, name)

    return cond_dist


def _prepare_loc_scale(loc, scale,
                       loc_layer_norm=False,
                       scale_shift=0.0, scale_identical=False):
    if loc_layer_norm:
        loc = tf.nn.batch_normalization(
            loc, *tf.nn.moments(loc, axes=[-1], keep_dims=True),
            offset=None, scale=None, variance_epsilon=1e-5
        )
    if scale_identical:
        scale = tf.math.add(
            tf.zeros_like(scale),
            tf.math.reduce_mean(scale, axis=-1, keep_dims=True)
        )
    scale = tf.math.add(scale, scale_shift)
    return loc, scale


def _make_partial_loc_scale(type="normal", loc_layer_norm=False,
                            scale_shift=0.0, scale_identical=False):
    def fn(loc, scale, name):
        loc, scale = _prepare_loc_scale(
            loc, scale,
            loc_layer_norm, scale_shift, scale_identical
        )

        if type == "normal":
            return PartialLocScaleDist(
                loc, functools.partial(
                    tfd.MultivariateNormalDiag, scale_diag=scale,
                    allow_nan_stats=False, name=name
                )
            )
        elif type == "laplace":
            return PartialLocScaleDist(
                loc, functools.partial(
                    tfd.Laplace, scale=scale,
                    allow_nan_stats=False, name=name
                ),
                batch_ndims=1
            )
        elif type == "logistic":
            return PartialLocScaleDist(
                loc, functools.partial(
                    tfd.Logistic, scale=scale,
                    allow_nan_stats=False, name=name
                ),
                batch_ndims=1
            )
        elif type == "gumbel":
            return PartialLocScaleDist(
                loc, functools.partial(
                    tfd.Gumbel, scale=scale,
                    allow_nan_stats=False, name=name
                ),
                batch_ndims=1
            )
        else:
            raise ValueError("Unsupported loc-scale distribution: " + type)

    return fn


def _make_loc_scale(type="normal", loc_layer_norm=False,
                    scale_shift=0.0, scale_identical=False):
    def fn(loc, scale, name):
        loc, scale = _prepare_loc_scale(
            loc, scale,
            loc_layer_norm, scale_shift, scale_identical
        )
        if type == "normal":
            return tfd.MultivariateNormalDiag(
                loc=loc, scale_diag=scale,
                allow_nan_stats=False, name=name
            )
        elif type == "laplace":
            return tfd.Independent(tfd.Laplace(
                loc=loc, scale=scale,
                allow_nan_stats=False, name=name
            ), reinterpreted_batch_ndims=1)
        elif type == "logistic":
            return tfd.Independent(tfd.Logistic(
                loc=loc, scale=scale,
                allow_nan_stats=False, name=name
            ), reinterpreted_batch_ndims=1)
        elif type == "gumbel":
            return tfd.Independent(tfd.Gumbel(
                loc=loc, scale=scale,
                allow_nan_stats=False, name=name
            ), reinterpreted_batch_ndims=1)
        else:
            raise ValueError("Unsupported loc-scale distribution: " + type)

    return fn


def _make_neg_binomial():
    def neg_binomial(logcnt, logit, name):
        return tfd.Indepdent(tfd.NegativeBinomial(
            total_count=tf.math.exp(logcnt), logits=logit,
            validate_args=False, allow_nan_stats=False,
            name=name
        ), reinterpreted_batch_ndims=1)
    return neg_binomial


def _make_cond_loc_scale(mlp, dim_from, dim_to,
                         type="normal",
                         loc_activation="linear",
                         loc_layer_norm=False,
                         scale_activation="softplus",
                         scale_shift=0.0, scale_identical=False,
                         name="cond_loc_scale",
                         **kwargs):
    make_fn = functools.partial(_make_partial_loc_scale, type=type)
    return _make_cond_two_param_dist(
        mlp=mlp, dim_from=dim_from, dim_to=dim_to,
        make_dist=make_fn(
            loc_layer_norm=loc_layer_norm,
            scale_shift=scale_shift,
            scale_identical=scale_identical
        ),
        act_0=loc_activation, act_1=scale_activation,
        name=name, **kwargs
    )


def _make_cond_neg_binomial(mlp, dim_from, dim_to,
                            name="cond_neg_binomial", **kwargs):
    return _make_cond_two_param_dist(
        mlp=mlp, dim_from=dim_from, dim_to=dim_to,
        make_dist=_make_neg_binomial(),
        act_0="linear", act_1="linear",
        name=name, **kwargs
    )


def _make_cond_mix_two_param_dist(
        mlp, num_components,
        dim_from, dim_to, make_dist,
        act_0="linear", act_1="linear", layer_norm=False,
        name="cond_mix_two_param_dist"):
    K = num_components
    components = []

    with tf.variable_scope(name):
        layer_norm_fn = tf.identity if not layer_norm \
            else layer_norm_1d(dim_from, trainable=True)
        hid_to_mix_weights = keras.Sequential([
            keras.layers.Activation(HID_ACTIVATION, input_shape=(dim_from,)),
            keras.layers.Dense(K, use_bias=False)
        ])
        for i in range(K):
            to_param_0 = _new_layer(
                dim_i=dim_from, dim_o=dim_to,
                act_i=HID_ACTIVATION, act_o=act_0,
                name="hid_to_param_0"
            )
            to_param_1 = _new_layer(
                dim_i=dim_from, dim_o=dim_to,
                act_i=HID_ACTIVATION, act_o=act_1,
                name="hid_to_param_1"
            )
            components.append((to_param_0, to_param_1))

    def cond_dist(input):
        hidden = layer_norm_fn(mlp(input))
        mix_weights = tf.math.softmax(tf.math.divide(
            hid_to_mix_weights(hidden),
            tf.math.sqrt(tf.cast(dim_from, tf.float32))
        ))
        param_0_list, param_1_list = [], []

        for i in range(K):
            hid_to_param_0, hid_to_param_1 = components[i]
            param_0 = hid_to_param_0(hidden)
            param_1 = hid_to_param_1(hidden)
            param_0_list.append(param_0)
            param_1_list.append(param_1)

        param_0s = tf.stack(param_0_list, axis=-2)
        param_1s = tf.stack(param_1_list, axis=-2)

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mix_weights),
            components_distribution=make_dist(
                param_0s, param_1s, name + "_components"
            ),
            name=name
        )

    return cond_dist


def _make_cond_mix_loc_scale(mlp, num_components, dim_from, dim_to,
                             type="normal",
                             loc_activation="linear",
                             loc_layer_norm=False,
                             scale_activation="softplus",
                             scale_shift=1e-4, scale_identical=False,
                             name="cond_mix_loc_scale",
                             **kwargs):
    return _make_cond_mix_two_param_dist(
        mlp=mlp, num_components=num_components,
        dim_from=dim_from, dim_to=dim_to,
        make_dist=_make_loc_scale(
            type=type,
            loc_layer_norm=loc_layer_norm,
            scale_shift=scale_shift,
            scale_identical=scale_identical
        ),
        act_0=loc_activation, act_1=scale_activation,
        name=name, **kwargs
    )


def _make_cond_mix_neg_binomial(mlp, num_components, dim_from, dim_to,
                                name="cond_mix_neg_binomial", **kwargs):
    return _make_cond_mix_two_param_dist(
        mlp=mlp, num_components=num_components,
        dim_from=dim_from, dim_to=dim_to,
        make_dist=_make_neg_binomial(),
        act_0="linear", act_1="linear",
        name=name, **kwargs
    )


def _make_mlp_layers(dim_in, dim_hid, dim_out, num_layers):
    if num_layers == 0 and dim_in == dim_out:
        return tf.identity
    if num_layers == 1:
        return mlp_one_layer(dim_in, dim_out)
    elif num_layers == 2:
        return mlp_two_layers(dim_in, dim_hid, dim_out)
    else:
        raise ValueError("Not supported: #layers = %d" % num_layers)


def _make_mlp_two_param_dist(make_dist):
    def make_mlp_dist(dim_in, dim_hid, dim_out, mlp_num_layers=1,
                      name="mlp_two_param_dist", **kwargs):
        dim_mlp_out = dim_in if mlp_num_layers <= 0 else dim_hid

        with tf.variable_scope(name):
            mlp = _make_mlp_layers(
                dim_in, dim_hid, dim_mlp_out, mlp_num_layers
            )
        return make_dist(
            mlp=mlp, dim_from=dim_mlp_out, dim_to=dim_out,
            name=name, **kwargs
        )

    return make_mlp_dist


def get_mlp_loc_scale_builder(type):
    return _make_mlp_two_param_dist(
        functools.partial(_make_cond_loc_scale, type=type)
    )


mlp_diag_normal = get_mlp_loc_scale_builder("normal")
mlp_laplace = get_mlp_loc_scale_builder("laplace")
mlp_logistic = get_mlp_loc_scale_builder("logistic")
mlp_neg_binomial = _make_mlp_two_param_dist(_make_cond_neg_binomial)


def _make_mlp_mix_two_param_dist(make_dist, fallback):
    def make_mlp_mix_dist(dim_in, dim_hid, dim_out,
                          mlp_num_layers=1, mix_num_components=1,
                          name="mlp_mix_two_param_dist", **kwargs):
        if mix_num_components == 1:
            return fallback(
                dim_in=dim_in, dim_hid=dim_hid, dim_out=dim_out,
                mlp_num_layers=mlp_num_layers, name=name,
                **kwargs
            )

        dim_mlp_out = dim_in if mlp_num_layers <= 0 else dim_hid

        with tf.variable_scope(name):
            mlp = _make_mlp_layers(
                dim_in, dim_hid, dim_mlp_out, mlp_num_layers
            )
        return make_dist(
            mlp=mlp, num_components=mix_num_components,
            dim_from=dim_mlp_out, dim_to=dim_out,
            name=name, **kwargs
        )

    return make_mlp_mix_dist


def get_mlp_mix_loc_scale_builder(type):
    return _make_mlp_mix_two_param_dist(
        functools.partial(_make_cond_mix_loc_scale, type=type),
        get_mlp_loc_scale_builder(type)
    )


mlp_mix_diag_normal = get_mlp_mix_loc_scale_builder("normal")
mlp_mix_laplace = get_mlp_mix_loc_scale_builder("laplace")
mlp_mix_logistic = get_mlp_mix_loc_scale_builder("logistic")
mlp_mix_neg_binomial = _make_mlp_mix_two_param_dist(
    _make_cond_mix_neg_binomial,
    mlp_neg_binomial
)


def identity_diag_normal(dim_in, dim_out,
                         scale_activation="softplus", scale_shift=0.0,
                         name="identity_diag_normal"):
    assert dim_in >= dim_out

    # identity_transform = tf.concat([
    #     tf.eye(dim_out), tf.zeros([dim_in - dim_out, dim_out])
    # ], axis=0)

    with tf.variable_scope(name):
        linear_transform = tf.get_variable(
            "linear", shape=[dim_in, dim_out],
            trainable=True, initializer=tf.initializers.glorot_normal()
        )

    def call(input):
        # loc = tf.linalg.tensordot(input, identity_transform)
        loc = input[..., :dim_out]
        scale_diag = lookup_activation_fn(scale_activation)(
            tf.linalg.tensordot(input, linear_transform, axes=1)
        )
        scale_diag = tf.math.add(scale_diag, scale_shift)
        return tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag,
            validate_args=True, allow_nan_stats=False,
            name=name
        )

    return call


def mlp_low_rank_normal(dim_in, dim_hid, dim_out,
                        mlp_num_layers=1, cov_rank=2,
                        loc_activation="linear",
                        scale_activation="softplus", scale_shift=1e-4,
                        name="mlp_low_rank_normal"):
    K = cov_rank
    dim_mlp_out = dim_in if mlp_num_layers <= 0 else dim_hid

    def _linear_layer(name, dim_o=None):
        return _new_layer(
            dim_i=dim_mlp_out, dim_o=(dim_out if dim_o is None else dim_o),
            act_i="linear", act_o="linear",
            name=name
        )

    with tf.variable_scope(name):
        hid_to_loc = _linear_layer("hid_to_loc")
        hid_to_diag = _linear_layer("hid_to_diags")
        hid_to_perturb_diag = _linear_layer("hid_to_perturb_diags", dim_o=K)
        hid_to_perturb_factor = tf.get_variable(
            "hid_to_perturb_factors", [dim_mlp_out, dim_out, K],
            trainable=True, initializer=tf.initializers.glorot_normal()
        )
        mlp = _make_mlp_layers(dim_in, dim_hid, dim_hid, mlp_num_layers)

    act_fn = lookup_activation_fn

    def call(input):
        hidden = mlp(input)
        hidden = act_fn(HID_ACTIVATION)(hidden)
        loc = act_fn(loc_activation)(hid_to_loc(hidden))
        scale_diag = act_fn(scale_activation)(hid_to_diag(hidden))
        scale_diag = tf.math.add(scale_diag, scale_shift)
        scale_perturb_diag = tf.math.softplus(hid_to_perturb_diag(hidden))
        scale_perturb_factor = tf.linalg.tensordot(
            hidden, hid_to_perturb_factor, axes=1
        )
        return PartialLocScaleDist(
            loc, functools.partial(
                tfd.MultivariateNormalDiagPlusLowRank,
                scale_diag=scale_diag,
                scale_perturb_diag=scale_perturb_diag,
                scale_perturb_factor=scale_perturb_factor,
                name=name
            )
        )

    return call


def layer_norm_1d(num_dims,
                  trainable=False, use_bias=False, eps=0.00001,
                  name="LayerNorm1D"):
    with tf.name_scope(name):
        gamma = tf.ones([num_dims], name="gamma")
        beta = tf.zeros([num_dims], name="beta")
        bias = tf.zeros([num_dims], name="bias")

    with tf.variable_scope(name):
        if trainable:
            gamma = tf.get_variable(
                "gamma", shape=[num_dims], trainable=True,
                initializer=tf.initializers.ones()
            )
            beta = tf.get_variable(
                "beta", shape=[num_dims], trainable=True,
                initializer=tf.initializers.zeros()
            )
        if use_bias:
            bias = tf.get_variable(
                "bias", shape=[num_dims], trainable=True,
                initializer=tf.initializers.glorot_normal()
            )

    def call(x):
        x = tf.math.add(x, bias)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = keras.backend.std(x, axis=-1, keepdims=True)
        normalized = tf.math.add(
            tf.multiply(
                gamma,  # TODO: `tf.div_no_nan` ?
                tf.math.divide(tf.subtract(x, mean), tf.math.add(std, eps))
            ), beta
        )
        normalized.set_shape(x.shape)
        return normalized

    return call


def gated_unit(dim_i, dim_o,
               gate_act=tf.math.sigmoid, value_act=tf.math.tanh,
               layer_norm=False, use_bias=True,
               name="gated_unit"):
    initializer = tf.initializers.glorot_normal
    with tf.variable_scope(name):
        gate_transform = tf.get_variable(
            "gate_transform", shape=[dim_i, dim_o],
            trainable=True, initializer=initializer()
        )
        value_transform = tf.get_variable(
            "value_transform", shape=[dim_i, dim_o],
            trainable=True, initializer=initializer()
        )
        gate_bias = value_bias = tf.zeros([dim_o])
        if use_bias:
            gate_bias = tf.get_variable(
                "gate_bias", shape=[dim_o],
                trainable=True, initializer=tf.initializers.zeros()
            )
            value_bias = tf.get_variable(
                "value_bias", shape=[dim_o],
                trainable=True, initializer=tf.initializers.zeros()
            )
        layer_norm_fn = tf.identity if not layer_norm \
            else layer_norm_1d(dim_o, trainable=True)

    def call(input):
        gate = tf.math.add(
            tf.linalg.tensordot(input, gate_transform, axes=1), gate_bias
        )
        value = tf.math.add(
            tf.linalg.tensordot(input, value_transform, axes=1), value_bias
        )
        return layer_norm_fn(
            tf.math.multiply(gate_act(gate), value_act(value))
        )

    return call


def cond_gated_unit(dim_i0, dim_i1, dim_o,
                    gate_act=tf.math.sigmoid, value_act=tf.math.tanh,
                    use_bias=True, layer_norm=False,
                    name="cond_gated_unit"):
    initializer = tf.initializers.glorot_normal
    with tf.variable_scope(name):
        gate_transform_0 = tf.get_variable(
            "gate_transform_0", shape=[dim_i0, dim_o],
            trainable=True, initializer=initializer()
        )
        gate_transform_1 = tf.get_variable(
            "gate_transform_1", shape=[dim_i1, dim_o],
            trainable=True, initializer=initializer()
        )
        value_transform_0 = tf.get_variable(
            "value_transform_0", shape=[dim_i0, dim_o],
            trainable=True, initializer=initializer()
        )
        value_transform_1 = tf.get_variable(
            "value_transform_1", shape=[dim_i1, dim_o],
            trainable=True, initializer=initializer()
        )
        gate_bias = value_bias = tf.zeros([dim_o])
        if use_bias:
            gate_bias = tf.get_variable(
                "gate_bias", shape=[dim_o],
                trainable=True, initializer=tf.initializers.zeros()
            )
            value_bias = tf.get_variable(
                "value_bias", shape=[dim_o],
                trainable=True, initializer=tf.initializers.zeros()
            )
        layer_norm_fn = tf.identity if not layer_norm \
            else layer_norm_1d(dim_o, trainable=True)

    def call(input_0, input_1):
        gate = tf.math.add(
            tf.linalg.tensordot(input_0, gate_transform_0, axes=1),
            tf.linalg.tensordot(input_1, gate_transform_1, axes=1)
        )
        value = tf.math.add(
            tf.linalg.tensordot(input_0, value_transform_0, axes=1),
            tf.linalg.tensordot(input_1, value_transform_1, axes=1)
        )
        return layer_norm_fn(tf.math.multiply(
            gate_act(tf.math.add(gate, gate_bias)),
            value_act(tf.math.add(value, value_bias))
        ))

    return call


def skip_cond_gated_unit(dim_i0, dim_i1, layer_norm=True,
                         name="skip_cond_gated_unit", **kwargs):
    with tf.variable_scope(name):
        _cond_gated = cond_gated_unit(
            dim_i0, dim_i1, dim_i0,
            layer_norm=False, **kwargs
        )
        transform = tf.get_variable(
            "transform", shape=[dim_i0, dim_i0],
            trainable=True, initializer=initializer()
        )
        layer_norm_fn = tf.identity if not layer_norm \
            else layer_norm_1d(dim_i0, trainable=True)

    def call(input_0, input_1):
        return layer_norm_fn(
            tf.math.add(input_0, tf.linalg.tensordot(
                _cond_gated(input_0, input_1), transform, axes=1
            ))
        )

    return call


def gated_linear_adder(dim_i, dim_g, name="gated_linear_adder"):
    with tf.variable_scope(name):
        gate_transform = mlp_two_layers(
            dim_in=dim_g,
            dim_hid=(4 * dim_i),
            dim_out=(2 * dim_i),
            act_out="sigmoid",
            name="gate_mlp_two_layers"
        )

    def call(oprand_0, oprand_1, controller):
        gate0, gate1 = tf.split(gate_transform(controller), 2, axis=-1)
        return tf.math.add(
            tf.math.multiply(gate0, oprand_0),
            tf.math.multiply(gate1, oprand_1)
        )

    return call


# https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/MatvecLU
def trainable_lu_factor(
        event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
    with tf.variable_scope(name, 'trainable_lu_factor', [event_size]):
        event_size = tf.convert_to_tensor(
            event_size, preferred_dtype=tf.int32, name='event_size'
        )
        random_matrix = tf.random_uniform(
            shape=tf.stack([event_size, event_size]),
            dtype=dtype, seed=seed
        )
        random_orthonormal = tf.linalg.qr(random_matrix)[0]
        lower_upper, permutation = tf.linalg.lu(random_orthonormal)
        lower_upper = tf.Variable(
            initial_value=lower_upper,
            trainable=True,
            use_resource=True,
            name='lower_upper'
        )
    return lower_upper, permutation


def trainable_qr_factor(event_size, k=None, eps=0.01,
                        seed=None, dtype=tf.float32, name=None):
    k = k or event_size
    assert event_size > 0 and event_size % 2 == 0
    assert k > 0

    with tf.variable_scope(name, 'trainable_qr_factor'):
        random_matrix = tf.random.uniform(
            shape=tf.stack([event_size, event_size]),
            dtype=dtype, seed=seed
        )
        rand_orthonormal, rand_upper_tril = tf.linalg.qr(random_matrix)

        flat_rand_upper_tril = tfd.fill_triangular_inverse(
            rand_upper_tril, upper=True
        )
        flat_upper_tril_var = tf.Variable(
            initial_value=flat_rand_upper_tril,
            trainable=True,
            use_resource=True,
            name='flat_upper_tril'
        )
        diag_var = tf.get_variable(
            name="diag_var", shape=[event_size],
            trainable=True, initializer=keras.initializers.zeros()
        )
        upper_tril = tfd.fill_triangular(flat_upper_tril_var, upper=True)
        upper_tril = tf.linalg.set_diag(upper_tril, exptanh(diag_var))

        orthonormal_var = tf.Variable(
            initial_value=rand_orthonormal[:k],
            trainable=True,
            use_resource=True,
            name='orthonormal_var'
        )
        orthogonal_matrix = tf.eye(event_size)
        for i in range(k):
            vi = orthonormal_var[i]
            Qi = tf.math.subtract(
                tf.eye(event_size),
                2.0 * tf.math.divide(
                    tf.linalg.einsum('i,j->ij', vi, vi),
                    tf.math.reduce_sum(tf.math.square(vi))
                )
            )
            orthogonal_matrix = tf.linalg.matmul(orthogonal_matrix, Qi)

        return orthogonal_matrix, upper_tril


# COPY FROM
#   https://github.com/microsoft/dpu-utils/blob/master/python/dpu_utils/tfutils/unsortedsegmentops.py
def unsorted_segment_log_softmax(logits, segment_ids, num_segments):
    '''
    Args:
      logits: A (D) Tensor.
      segment_ids: A (D) Tensor.

    Returns:
      log_probs: A (D) Tensor.
    '''
    max_per_segment = tf.unsorted_segment_max(
        data=logits,
        segment_ids=segment_ids, num_segments=num_segments
    )
    scattered_maxes = tf.gather(params=max_per_segment, indices=segment_ids)
    recentered_scores = logits - scattered_maxes
    exped_recentered_scores = tf.exp(recentered_scores)

    per_segment_sums = tf.unsorted_segment_sum(
        exped_recentered_scores, segment_ids, num_segments)
    per_segment_normalization_consts = tf.log(per_segment_sums)

    log_probs = recentered_scores - \
        tf.gather(params=per_segment_normalization_consts, indices=segment_ids)
    return log_probs
