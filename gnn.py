from __future__ import absolute_import
from __future__ import print_function

from util import layer_norm_1d, lookup_activation_fn
import util

import functools
import copy

import tensorflow as tf
from tensorflow import keras

DOT = functools.partial(tf.linalg.tensordot, axes=1)

GNN_SPARSE = "sparse"
GNN_DENSE = "dense"
GNN_DISABLE = "disable"

ATTENTION_SOFTMAX = "softmax"
ATTENTION_SIGMOID = "sigmoid"
ATTENTION_UNIFORM = "uniform"

MESSENGER_BINARY = "binary"
MESSENGER_UNARY = "unary"

COMBINER_LSTM = "lstm"
COMBINER_GRU = "gru"
COMBINER_ADD = "add"

# READOUT_MEAN = "mean"
# READOUT_MAX = "max"
READOUT_DISABLE = "disable"
READOUT_MEAN_MAX = "mean_max"


class MultiHeadAttConfig(object):
    def __init__(self, num_heads, dim_input, dim_key, dim_value,
                 dim_node_attr=None, dim_edge_attr=None,
                 dim_global_state=None,
                 impl=GNN_SPARSE, attention=ATTENTION_SOFTMAX,
                 messenger=MESSENGER_BINARY,
                 layer_norm_in=True, layer_norm_out=False,
                 skip_conn=False, feed_forward=False,
                 activation="linear",
                 parallel_iterations=10, swap_memory=False):
        self._num_heads = num_heads
        self._dim_input = dim_input
        self._dim_key = dim_key
        self._dim_value = dim_value
        self._dim_node_attr = 0 if dim_node_attr is None else dim_node_attr
        self._dim_edge_attr = 0 if dim_edge_attr is None else dim_edge_attr
        self._dim_global_state = 0 \
            if dim_global_state is None else dim_global_state
        self._impl = impl
        self._attention = attention
        self._messenger = messenger
        self._layer_norm_in = layer_norm_in
        self._layer_norm_out = layer_norm_out
        self._skip_conn = skip_conn
        self._activation = activation
        self._feed_forward = feed_forward
        self._feed_forward_act = activation
        self._parallel_iterations = parallel_iterations
        self._swap_memory = swap_memory

    def clone(self):
        return copy.deepcopy(self)

    @property
    def num_heads(self):
        return self._num_heads

    @num_heads.setter
    def num_heads(self, value):
        self._num_heads = value

    @property
    def dim_input(self):
        return self._dim_input

    @dim_input.setter
    def dim_input(self, value):
        self._dim_input = value

    @property
    def dim_key(self):
        return self._dim_key

    @dim_key.setter
    def dim_key(self, value):
        self._dim_key = value

    @property
    def dim_value(self):
        return self._dim_value

    @dim_value.setter
    def dim_value(self, value):
        self._dim_value = value

    @property
    def dim_node_attr(self):
        return self._dim_node_attr

    @property
    def dim_edge_attr(self):
        return self._dim_edge_attr

    @property
    def dim_global_state(self):
        return self._dim_global_state

    @dim_global_state.setter
    def dim_global_state(self, value):
        self._dim_global_state = value

    @property
    def impl(self):
        return self._impl

    @property
    def attention(self):
        return self._attention

    @attention.setter
    def attention(self, value):
        self._attention = value

    @property
    def messenger(self):
        return self._messenger

    @messenger.setter
    def messenger(self, value):
        self._messenger = value

    @property
    def layer_norm_in(self):
        return self._layer_norm_in

    @layer_norm_in.setter
    def layer_norm_in(self, value):
        self._layer_norm_in = value

    @property
    def layer_norm_out(self):
        return self._layer_norm_out

    @layer_norm_out.setter
    def layer_norm_out(self, value):
        self._layer_norm_out = value

    @property
    def skip_conn(self):
        return self._skip_conn

    @skip_conn.setter
    def skip_conn(self, value):
        self._skip_conn = value

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        self._activation = value

    @property
    def feed_forward(self):
        return self._feed_forward

    @feed_forward.setter
    def feed_forward(self, value):
        self._feed_forward = value

    @property
    def feed_forward_act(self):
        return self._feed_forward_act

    @feed_forward_act.setter
    def feed_forward_act(self, value):
        self._feed_forward_act = value

    @property
    def parallel_iterations(self):
        return self._parallel_iterations

    @property
    def swap_memory(self):
        return self._swap_memory


class GNNConfig(MultiHeadAttConfig):
    def __init__(self, *args,
                 num_layers=2, recurrent=False,
                 combiner=COMBINER_LSTM, rnn_num_layers=1,
                 readout=None, **kwargs):
        super(GNNConfig, self).__init__(*args, **kwargs)
        assert (combiner != COMBINER_ADD) or (num_layers == 1)
        self._num_layers = num_layers
        self._combiner = combiner
        self._rnn_num_layers = rnn_num_layers
        self._recurrent = recurrent
        self._readout = READOUT_DISABLE if readout is None else readout

    def clone(self):
        return copy.deepcopy(self)

    @property
    def num_layers(self):
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value):
        self._num_layers = value

    @property
    def recurrent(self):
        return self._recurrent

    @property
    def combiner(self):
        return self._combiner

    @combiner.setter
    def combiner(self, value):
        self._combiner = value

    @property
    def readout(self):
        return self._readout

    @readout.setter
    def readout(self, value):
        self._readout = value or READOUT_DISABLE

    @property
    def dim_readout(self):
        if self.readout == READOUT_MEAN_MAX:
            return self.dim_input
        elif self.readout == READOUT_DISABLE:
            return 0
        raise ValueError("Unknown readout function: " + self.readout)

    @property
    def rnn_num_layers(self):
        return self._rnn_num_layers

    @property
    def dim_full_state(self):
        if self.combiner == COMBINER_LSTM:
            return 2 * self.rnn_num_layers * self.dim_input
        elif self.combiner == COMBINER_GRU:
            return self.rnn_num_layers * self.dim_input
        return self.dim_input

    @property
    def single_layer(self):
        return super(GNNConfig, self).clone()


###########################################################
#                     Attention Methods                   #
###########################################################

def _sparse_attention_softmax_fast(config, graph, similarities,
                                   reverse_mask=None):
    '''
    Normalize the similarities with softmax.
    Only work in the `graph signal` mode, i.e., the graphs
    in a mini-batch are identical.

    Args:
      similarities: A (..., B, E) Tensor.
      reverse_mask: Optional. A ([..., ]B, E) Tensor.

    Returns:
      normalized_weights: A (..., B, E) Tensor.
    '''
    full_prefix = tf.shape(similarities)[:-1]

    similarities = tf.math.add(similarities, tf.math.multiply(
        util.SOFTMAX_MASK_MULTIPLIER, (1.0 - graph.edge_mask)
    ))
    if reverse_mask is not None:
        similarities = tf.math.add(similarities, tf.math.multiply(
            util.SOFTMAX_MASK_MULTIPLIER, util.float(reverse_mask)
        ))

    # [N1, N2, NB] -> [0, N1, N1+N2, ...]
    # (B) -> (B, 1) +-> (B, E)
    offsets = tf.math.cumsum(graph.num_nodes, exclusive=True)
    offsets = tf.math.add(
        tf.expand_dims(offsets, axis=-1),
        tf.zeros(tf.shape(graph.edges)[:-1], dtype=tf.int32)
    )

    # (...) -> (F)
    extra_batch_size = tf.math.reduce_prod(full_prefix[:-1])
    extra_offsets = tf.math.multiply(
        tf.range(extra_batch_size), graph.total_num_nodes
    )

    # (B, E) + (F, 1, 1) -> (F, B, E)
    #                    +  (B, E) -> (F, B, E)
    offsets = tf.math.add(offsets, util.append_dims(extra_offsets, 2))
    relabeled_receivers = tf.math.add(offsets, graph.edges[..., 1])

    # (..., B, E) -> (... * B * E)
    flat_similarities = tf.reshape(similarities, [-1])
    flat_segment_ids = tf.reshape(relabeled_receivers, [-1])

    flat_normalized_scores = tf.exp(util.unsorted_segment_log_softmax(
        logits=flat_similarities, segment_ids=flat_segment_ids,
        num_segments=tf.math.multiply(extra_batch_size, graph.total_num_nodes)
    ))
    return tf.reshape(flat_normalized_scores, tf.shape(similarities))


def _sparse_attention_softmax_slow(config, graph, similarities,
                                   reverse_mask=None):
    '''
    Normalize the similarities with softmax.
    - This function works properly even if the graphs
      in a mini-batch are different.
    - It is very slow on GPU, because TF creates `SparseTensor`s
      on CPU.

    Args:
      similarities: A (..., B, E) Tensor.
      reverse_mask: Optional. A ([..., ]B, E) Tensor.

    Returns:
      normalized_weights: A (..., B, E) Tensor.
    '''
    full_prefix = tf.shape(similarities)[:-1]
    batch_size = full_prefix[-1]
    max_num_edges = graph.max_num_edges

    if reverse_mask is not None:
        similarities = tf.math.add(similarities, tf.math.multiply(
            util.SOFTMAX_MASK_MULTIPLIER, util.float(reverse_mask)
        ))

    # (..., B, E) -> (...*B, E)
    flat_similarities = tf.reshape(
        similarities,
        tf.stack([tf.math.reduce_prod(full_prefix), max_num_edges])
    )
    # (B) -> (..., B) -> (...*B)
    batch_ids = tf.math.add(
        tf.range(batch_size),
        tf.zeros(full_prefix, dtype=tf.int32)
    )
    flat_batch_ids = tf.reshape(
        batch_ids, tf.stack([tf.math.reduce_prod(full_prefix)])
    )

    def sparse_softmax(params):
        batch_similarities, batch_id = params
        max_num_edges = tf.shape(batch_similarities)[0]
        num_nodes = tf.cast(graph.num_nodes[batch_id], tf.int64)
        num_edges = graph.num_edges[batch_id]
        adj_shape = tf.stack([num_nodes, num_nodes])
        sparse_similarities = tf.SparseTensor(
            indices=graph.edges_int64[batch_id][:num_edges],
            values=batch_similarities[:num_edges],
            dense_shape=adj_shape
        )
        # For directed graph, the adjacency matrix is indexed by [sid,rid].
        # However `sparse.softmax` will normalize weights along the last dim,
        # so we first transpose the weight matrix to be indexed by [rid,sid]
        # before applying `sparse.softmax`, then transpose the normalized
        # matrix back to be indexed by [sid,rid].
        sparse_weights = tf.sparse.transpose(
            tf.sparse.softmax(tf.sparse.transpose(sparse_similarities))
        )
        return tf.concat([
            sparse_weights.values, tf.zeros(max_num_edges - num_edges)
        ], axis=-1)

    # NOTE:
    #
    # The `while_loop` in `map_fn` makes this implementation
    # not twice-differentiable, because TensorFlow will raise
    # "TypeError: Second-order gradient for while loops not supported."
    #
    flat_normalized_weights = tf.map_fn(
        sparse_softmax, (flat_similarities, flat_batch_ids),
        dtype=tf.float32,
        parallel_iterations=config.parallel_iterations,
        swap_memory=config.swap_memory
    )
    normalized_weights = tf.reshape(
        flat_normalized_weights, shape=tf.shape(similarities)
    )
    return normalized_weights


def _sparse_attention_softmax_slow_alt(config, graph, similarities,
                                       reverse_mask=None):
    del config
    if reverse_mask is not None:
        similarities = tf.math.add(similarities, tf.math.multiply(
            util.SOFTMAX_MASK_MULTIPLIER, util.float(reverse_mask)
        ))

    sparse_similarities = graph.dense_edge_weights_to_sparse(similarities)
    perm = tf.range(sparse_similarities.shape.ndims)
    perm = tf.stack([*tf.unstack(perm[:-2]), perm[-1], perm[-2]])
    sparse_normalized_weights = tf.sparse.transpose(tf.sparse.softmax(
        tf.sparse.transpose(sparse_similarities, perm)
    ), perm)
    return graph.sparse_edge_weights_to_dense(sparse_normalized_weights)


def _sparse_attention_sigmoid(config, graph, similarities, reverse_mask=None):
    del config, reverse_mask
    tf.math.multiply(graph.edge_mask, tf.math.sigmoid(similarities))


def _sparse_attention_uniform(config, graph, similarities, reverse_mask=None):
    del config
    ones = tf.math.multiply(tf.ones_like(similarities), graph.edge_mask)
    if reverse_mask is not None:  # TODO
        ones = tf.math.multiply(ones, (1.0 - util.float(reverse_mask)))
    return tf.math.xdivy(ones, graph.tail_indegree)


def _dense_attention_softmax(config, graph, similarities, reverse_mask=None):
    '''
    Args:
      similarities: A (..., B, N, N) Tensor indexed by (sid, rid).
      reverse_mask: Optional. A (..., B, N, N) Tensor indexed by (sid, rid).

    Returns:
      normalized_weights: A (..., B, N, N) Tensor indexed by (rid, sid).
    '''
    del config

    mask = graph.gen_adj_mask_like(
        similarities, reverse_mask=reverse_mask, transpose=True
    )
    # (..., B, N[r], N[s])
    #
    # MASKED SOFTMAX
    #
    # Slow Implementation for (N, N):
    #
    # indices = tf.where(mask)
    # compats_sparse = tf.SparseTensor(
    #     indices, tf.gather_nd(similarities, indices),
    #     tf.shape(similarities, out_type=tf.int64)
    # )
    # att_weights_sparse = tf.sparse_softmax(compats_sparse)
    # att_weights = tf.sparse.to_dense(att_weights_sparse)
    #
    # Dirty but Fast Implementation:
    #
    # e.g., see:
    # https://github.com/google-research/bert/blob/master/modeling.py
    # https://github.com/tensorflow/tensorflow/issues/11756
    #
    mask_multiplier = util.float(mask)
    reverse_mask_adder = tf.math.multiply(
        1.0 - mask_multiplier, tf.constant(util.SOFTMAX_MASK_MULTIPLIER)
    )
    weights = tf.math.add(similarities, reverse_mask_adder)
    normalized_weights = tf.math.softmax(weights, axis=-1)
    return tf.math.multiply(mask_multiplier, normalized_weights)


def _dense_attention_sigmoid(config, graph, similarities, reverse_mask=None):
    del config

    mask = graph.gen_adj_mask_like(
        similarities, reverse_mask=reverse_mask, transpose=True
    )
    mask_multiplier = util.float(mask)
    normalized_weights = tf.math.sigmoid(similarities)
    return tf.math.multiply(normalized_weights, mask_multiplier)


def _dense_attention_uniform(config, graph, similarities, reverse_mask=None):
    del config

    mask = graph.gen_adj_mask_like(
        similarities, reverse_mask=reverse_mask, transpose=True
    )
    mask_multiplier = util.float(mask)
    uniform_similarities = tf.math.multiply(
        tf.ones_like(similarities), mask_multiplier
    )
    recv_indegree = tf.math.reduce_sum(mask_multiplier, axis=-1)
    recv_indegree = tf.expand_dims(recv_indegree, axis=-1)
    normalized_weights = tf.div_no_nan(uniform_similarities, recv_indegree)
    return normalized_weights


_sparse_attention_methods = {
    ATTENTION_SOFTMAX: _sparse_attention_softmax_fast,
    ATTENTION_SIGMOID: _sparse_attention_sigmoid,
    ATTENTION_UNIFORM: _sparse_attention_uniform
}

_dense_attention_methods = {
    ATTENTION_SOFTMAX: _dense_attention_softmax,
    ATTENTION_SIGMOID: _dense_attention_sigmoid,
    ATTENTION_UNIFORM: _dense_attention_uniform
}


def _lookup_attention_method(config):
    methods = _sparse_attention_methods if config.impl == GNN_SPARSE \
        else _dense_attention_methods
    attention = methods[config.attention]
    if attention is None:
        raise ValueError("unknown attention method: " + config.attention)
    return functools.partial(attention, config)


###########################################################
#                         Messengers                      #
###########################################################


class BinaryMessenger(object):
    def __init__(self, num_heads,
                 dim_input, dim_global_state, dim_edge_attr, dim_msg,
                 activation=tf.tanh, name="BinaryMessenger"):
        def var(name, dim_i):
            return tf.get_variable(
                name, shape=[dim_i, num_heads, dim_msg], trainable=True,
                initializer=tf.initializers.glorot_normal()
            )

        with tf.variable_scope(name):
            self._send_to_gate = var("send_to_gate", dim_input)
            self._recv_to_gate = var("recv_to_gate", dim_input)
            self._send_to_effect = var("send_to_effect", dim_input)
            self._recv_to_effect = var("recv_to_effect", dim_input)
            self._edge_attr_to_gate = var(
                "edge_attr_to_gate", dim_edge_attr)
            self._edge_attr_to_effect = var(
                "edge_attr_to_effect", dim_edge_attr)
            self._global_state_to_gate = var(
                "global_state_to_gate", dim_global_state)
            self._global_state_to_effect = var(
                "global_state_to_effect", dim_global_state)

        self._dim_edge_attr = dim_edge_attr
        self._dim_global_state = dim_global_state
        self._activation = activation

    def _transform_edge_attrs(self, edge_attrs, expand_axis, times):
        assert (edge_attrs is None) == (self._dim_edge_attr <= 0)

        if edge_attrs is None:
            return 0.0, 0.0

        def expand(x):
            for _ in range(times):
                x = tf.expand_dims(x, axis=expand_axis)
            return x

        # (B,E,de) * (de,nh,dm) -> (B,E,nh,dm) -> (B,E,...,nh,dm)
        # (B,N,N,de) * (de,nh,dm) -> (B,N,N,nh,dm) -> (...,B,M,N,nh,dm)
        return (
            expand(DOT(edge_attrs, self._edge_attr_to_gate)),
            expand(DOT(edge_attrs, self._edge_attr_to_effect))
        )

    def _transform_global_states(self, global_states, expand_axis, times):
        assert (global_states is None) == (self._dim_global_state <= 0)

        if global_states is None:
            return 0.0, 0.0

        def expand(x):
            for _ in range(times):
                x = tf.expand_dims(x, axis=expand_axis)
            return x

        return (
            expand(DOT(global_states, self._global_state_to_gate)),
            expand(DOT(global_states, self._global_state_to_effect))
        )

    def compute_message_sparse(self, graph, senders, receivers,
                               global_states=None):
        '''
        Args:
          senders: A (..., B, N, d) Tensor.
          receivers: A (..., B, M, d) Tensor.
          global_states: Optional. A (..., B, dg) Tensor.

        Returns:
          messages: A (..., nh, B, E, dm) Tensor.
        '''
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(senders)[:-2], tf.shape(receivers)[:-2]),
            tf.assert_equal(tf.shape(senders)[-1], tf.shape(receivers)[-1])
        ]):
            sids, rids = graph.edges[..., 0], graph.edges[..., 1]
            unknown_prefix_length = len(senders.shape.as_list()[:-3])

        # (...,B,N,d) * (d,nh,dm) -> (...,B,N,nh,dm) -> (B,N,...,nh,dm)
        perm = tf.range(senders.shape.ndims + 1)
        # (...,-4,-3,-2,-1) -> (-4,-3,...,-2,-1)
        perm = tf.stack([
            perm[-4], perm[-3], *tf.unstack(perm[:-4]), perm[-2], perm[-1]
        ])
        as_gate_send = tf.transpose(
            DOT(senders, self._send_to_gate), perm)
        as_gate_recv = tf.transpose(
            DOT(receivers, self._recv_to_gate), perm)
        as_effect_send = tf.transpose(
            DOT(senders, self._send_to_effect), perm)
        as_effect_recv = tf.transpose(
            DOT(receivers, self._recv_to_effect), perm)

        # (B,N,...,nh,dm) -> (B,E,...,nh,dm)
        gate_send = tf.batch_gather(as_gate_send, indices=sids)
        effect_send = tf.batch_gather(as_effect_send, indices=sids)
        gate_recv = tf.batch_gather(as_gate_recv, indices=rids)
        effect_recv = tf.batch_gather(as_effect_recv, indices=rids)

        # (B,E,de) -> (B,E,...,nh,dm)
        gate_edge, effect_edge = self._transform_edge_attrs(
            edge_attrs=graph.edge_attrs,
            expand_axis=-3, times=unknown_prefix_length
        )

        # (...,B,dg) -> (B,...,dg)
        if global_states is not None:
            perm = tf.range(global_states.shape.ndims)
            # (...,-2,-1) -> (-2,...,-1)
            perm = tf.stack([perm[-2], *tf.unstack(perm[:-2]), perm[-1]])
            global_states = tf.transpose(global_states, perm)
        # (B,...,dg) -> (B,...,nh,dm) -> (B,1,...,nh,dm)
        gate_global, effect_global = self._transform_global_states(
            global_states=global_states, expand_axis=1, times=1
        )

        gates = tf.math.add(gate_send, gate_recv)
        gates = tf.math.add(gates, gate_edge)
        gates = tf.math.add(gates, gate_global)
        effects = tf.math.add(effect_send, effect_recv)
        effects = tf.math.add(effects, effect_edge)
        effects = tf.math.add(effects, effect_global)

        # (B,E,...,nh,dm) -> (...,nh,B,E,dm)
        messages = tf.math.multiply(
            tf.math.sigmoid(gates), self._activation(effects)
        )
        perm = tf.range(messages.shape.ndims)
        # (0,1,...,-2,-1) -> (...,-2,0,1,-1)
        perm = tf.stack([
            *tf.unstack(perm[2:-2]), perm[-2], perm[0], perm[1], perm[-1]
        ])
        messages = tf.transpose(messages, perm)

        # (...,nh,B,E,dm) * (B,E,1)
        return graph.mask_edge_info(messages, ndims=1)

    def compute_message_dense(self, graph, senders, receivers,
                              global_states=None):
        '''
        Args:
          senders: A (..., B, N, d) Tensor.
          receivers: A (..., B, M, d) Tensor.
          global_states: Optional. A (..., B, dg) Tensor.

        Returns:
          pairwise_messages: A (..., B, M, N, nh, dm) Tensor.
        '''
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(senders)[:-2], tf.shape(receivers)[:-2]),
            tf.assert_equal(tf.shape(senders)[-1], tf.shape(receivers)[-1])
        ]):
            unknown_prefix_length = len(senders.shape.as_list()[:-3])
            batch_shape = tf.shape(senders)[:-2]
            num_senders = tf.shape(senders)[-2]
            num_receivers = tf.shape(receivers)[-2]

        # (...,B,N,d) * (d,nh,dm) -> (...,B,N,nh,dm)
        gate_send = DOT(senders, self._send_to_gate)
        gate_recv = DOT(receivers, self._recv_to_gate)
        effect_send = DOT(senders, self._send_to_effect)
        effect_recv = DOT(receivers, self._recv_to_effect)

        # (...,B,1,N,nh,dm) -> (...,B,M,N,nh,dm)
        multiples_send = tf.stack([
            *tf.unstack(tf.ones(tf.size(batch_shape), dtype=tf.int32)),
            num_receivers, 1, 1, 1
        ])
        gate_send_expanded = tf.tile(
            tf.expand_dims(gate_send, axis=-4), multiples_send
        )
        effect_send_expanded = tf.tile(
            tf.expand_dims(effect_send, axis=-4), multiples_send
        )

        # (...,B,M,1,nh,dm) -> (...,B,M,N,nh,dm)
        multiples_recv = tf.stack([
            *tf.unstack(tf.ones(tf.size(batch_shape), dtype=tf.int32)),
            1, num_senders, 1, 1
        ])
        gate_recv_expanded = tf.tile(
            tf.expand_dims(gate_recv, axis=-3), multiples_recv
        )
        effect_recv_expanded = tf.tile(
            tf.expand_dims(effect_recv, axis=-3), multiples_recv
        )

        # (B,E,de) -> (...,B,M,N,nh,dm)
        gate_edge, effect_edge = self._transform_edge_attrs(
            edge_attrs=graph.dense_edge_attrs,
            expand_axis=0, times=unknown_prefix_length
        )
        # (...,B,dg) -> (...,B,nh,dm) -> (...,B,1,1,nh,dm)
        gate_global, effect_global = self._transform_global_states(
            global_states=global_states, expand_axis=-3, times=2
        )

        gates = tf.math.add(gate_send_expanded, gate_recv_expanded)
        gates = tf.math.add(gates, gate_edge)
        gates = tf.math.add(gates, gate_global)
        effects = tf.math.add(effect_send_expanded, effect_recv_expanded)
        effects = tf.math.add(effects, effect_edge)
        effects = tf.math.add(effects, effect_global)

        pairwise_messages = tf.math.multiply(
            tf.math.sigmoid(gates), self._activation(effects)
        )
        return pairwise_messages


class UnaryMessenger(object):
    def __init__(self, num_heads,
                 dim_input, dim_global_state, dim_edge_attr, dim_msg,
                 name="UnaryMessenger"):
        del dim_global_state, dim_edge_attr

        with tf.variable_scope(name):
            self._message_transform = tf.get_variable(
                "message_transform",
                shape=[dim_input, num_heads, dim_msg], trainable=True,
                initializer=tf.initializers.glorot_normal()
            )

    def compute_message_dense(self, graph, senders, receivers,
                              global_states=None):
        '''
        Args:
          graph: A RuntimeGraph object.
          senders: A (..., B, N, d) Tensor.
          receivers: A (..., B, M, d) Tensor.
          globals: Optional. A (..., B, dg) Tensor.

        Returns:
          messages: A (..., nh, B, N, dm) Tensor.
        '''
        del graph, receivers

        # (...,B,N,d) * (d,nh,dv) -> (...,B,N,nh,dv)
        messages = DOT(senders, self._message_transform)

        # global_bias = 0.0
        # if global_states is not None:
        #     # (...,B,dg) -> (...,B,nh,dv) -> (...,B,1,nh,dv)
        #     global_bias = DOT(global_states, self._global_state_transform)
        #     global_bias = tf.expand_dims(global_bias, axis=-3)
        # messages = tf.math.add(messages, global_bias)

        # (...,B,N,nh,dv) -> (...,nh,B,N,dv)
        perm = tf.range(messages.shape.ndims)
        # (...,-4,-3,-2,-1) -> (...,-2,-4,-3,-1)
        perm = tf.stack([
            *tf.unstack(perm[:-4]), perm[-2], perm[-4], perm[-3], perm[-1]
        ])
        return tf.transpose(messages, perm)

    def compute_message_sparse(self, graph, senders, receivers,
                               global_states=None):
        '''
        Args:
          graph: A RuntimeGraph object.
          senders: A (..., B, N, d) Tensor.
          receivers: A (..., B, M, d) Tensor.
          globals: Optional. A (..., B, dg) Tensor.

        Returns:
          messages: A (..., B, E, nh, dm) Tensor.
        '''
        del receivers

        sids = graph.edges[..., 0]

        # (...,B,N,d) * (d,nh,dm) -> (...,B,N,nh,dm) -> (B,N,...,nh,dm)
        perm = tf.range(senders.shape.ndims + 1)
        # (...,-4,-3,-2,-1) -> (-4,-3,...,-2,-1)
        perm = tf.stack([
            perm[-4], perm[-3], *tf.unstack(perm[:-4]), perm[-2], perm[-1]
        ])
        values = tf.transpose(DOT(senders, self._message_transform), perm)

        # global_bias = 0.0
        # if global_states is not None:
        #     # (...,B,dg) -> (...,B,1,dg)
        #     global_states = tf.expand_dims(global_states, axis=-2)
        #     # (...,B,1,nh,dm) -> (B,1,...,nh,dm)
        #     global_bias = tf.transpose(
        #         DOT(global_states, self._global_state_transform), perm
        #     )
        # values = tf.math.add(values, global_bias)

        # (B,N,...,nh,dm) -> (B,E,...,nh,dm) -> (...,nh,B,E,dm)
        messages = tf.batch_gather(values, indices=sids)
        # (0,1,...,-2,-1) -> (...,-2,0,1,-1)
        perm = tf.range(messages.shape.ndims)
        perm = tf.stack([
            *tf.unstack(perm[2:-2]), perm[-2], perm[0], perm[1], perm[-1]
        ])
        messages = tf.transpose(messages, perm)

        # (...,nh,B,E,dm) * (B,E,1)
        mask = tf.expand_dims(graph.edge_mask, axis=-1)
        return tf.math.multiply(messages, mask)


def _make_message_fn(config):
    ''' NOTE: call this function within a variable scope. '''
    params = dict(
        num_heads=config.num_heads,
        dim_input=(
            config.dim_input +
            config.dim_node_attr + config.dim_global_state
        ),
        dim_global_state=config.dim_global_state,
        dim_edge_attr=config.dim_edge_attr,
        dim_msg=config.dim_value
    )
    if config.messenger == MESSENGER_BINARY:
        messenger = BinaryMessenger(**params)
    elif config.messenger == MESSENGER_UNARY:
        messenger = UnaryMessenger(**params)
    else:
        raise ValueError("unknown messenger: " + config.messenger)

    if config.impl == GNN_SPARSE:
        return messenger.compute_message_sparse
    return messenger.compute_message_dense


###########################################################
#                   Graph Neural Networks                 #
###########################################################


def GraphNN(config, dim_out=None, name="GraphNN"):
    with tf.variable_scope(name + "Wrapper"):
        if config.feed_forward:
            feed_forward = util.mlp_two_layers(
                dim_in=config.dim_input,
                dim_hid=config.num_heads * config.dim_value,
                dim_out=(dim_out or config.dim_input),
                act_out=config.feed_forward_act,
                weight_init='small'
            )
        else:
            feed_forward = tf.identity

        if config.impl == GNN_SPARSE:
            _call = SparseGraphNN(config, name=name)
        elif config.impl == GNN_DENSE:
            _call = DenseGraphNN(config, name=name)
        elif config.impl == GNN_DISABLE:
            _call = DummyGraphNN(config, name=name)
        else:
            raise ValueError("unknown GNN implementation: " + config.impl)

    def call(*args, **kwargs):
        states = _call(*args, **kwargs)
        return feed_forward(states)

    return call


def _concat_global_states(states, global_states):
    '''
    Args:
      states: A (..., N, d) Tensor.
      global_states: A (..., dg) Tensor.

    Returns:
      concat_updates: A (..., N, d+dg) Tensor.
    '''
    if global_states is None:
        return states
    return util.broadcast_concat(
        states, tf.expand_dims(global_states, axis=-2)
    )


def _concat_node_attrs(graph, states, has_node_attr):
    assert not (has_node_attr and graph.node_attrs is None)
    if has_node_attr:
        states = util.broadcast_concat(states, graph.node_attrs)
    return states


def _make_layer_norm_fn(config):
    ''' NOTE: call this function within a variable scope. '''
    norm_in, norm_out = tf.identity, tf.identity
    if config.layer_norm_in:
        norm_in = layer_norm_1d(
            config.dim_input, trainable=True,
            name="InputLayerNorm"
        )
    if config.layer_norm_out:
        norm_out = layer_norm_1d(
            config.dim_input, trainable=True,
            name="OutputLayerNorm"
        )
    return norm_in, norm_out


def _make_skip_conn(config):
    def _fn(states, updates):
        if not config.skip_conn:
            return updates
        return tf.math.add(states, updates)
    return _fn


def DummyGraphNN(config, name="DummyGraphNN"):
    del config, name

    def call(graph, states, global_states=None):
        return tf.zeros_like(states)

    return call


def SparseGraphNN(config, name="SparseGraphNN"):
    num_heads = config.num_heads
    dim_input = config.dim_input
    dim_key = config.dim_key
    dim_value = config.dim_value
    dim_node_attr = config.dim_node_attr
    dim_edge_attr = config.dim_edge_attr
    dim_output = dim_input

    has_node_attr = (dim_node_attr > 0)
    has_edge_attr = (dim_edge_attr > 0)

    dim_input += (dim_node_attr + config.dim_global_state)

    ATTENTION = _lookup_attention_method(config)
    ACTIVATION = lookup_activation_fn(config.activation)
    SKIP_CONN = _make_skip_conn(config)

    with tf.variable_scope(name):
        initializer = tf.initializers.glorot_normal

        key_transform = tf.get_variable(
            "key_transform", shape=[dim_input, num_heads, dim_key],
            initializer=initializer(), trainable=True)
        query_transform = tf.get_variable(
            "query_transform", shape=[dim_input, num_heads, dim_key],
            initializer=initializer(), trainable=True)
        edge_attr_transform = tf.get_variable(
            "edge_attr_transform", shape=[dim_edge_attr, num_heads],
            initializer=initializer(), trainable=True)
        inverse_transform = tf.get_variable(
            "inverse_transform", shape=[num_heads, dim_value, dim_output],
            initializer=initializer(), trainable=True)
        head_transform = tf.get_variable(
            "head_transform", shape=[dim_input, num_heads],
            initializer=initializer(), trainable=True)

        MESSAGE = _make_message_fn(config)
        LAYER_NORM_I, LAYER_NORM_O = _make_layer_norm_fn(config)

    def call(graph, states, global_states=None, reverse_mask=None):
        '''
        Args:
          graph: A RuntimeGraph object.
          states: A (..., B, N, d) Tensor.
          global_states: Optional. A (..., B, d) Tensor.
          reverse_mask: Optional. A ([..., ]B, E) boolean Tensor.

        Returns:
          updates: A (..., B, N, d) Tensor.
        '''
        original_states = states
        states = LAYER_NORM_I(states)
        states = graph.mask_nodal_info(states)
        states = _concat_node_attrs(graph, states, has_node_attr)
        states = _concat_global_states(states, global_states)

        max_num_nodes = tf.shape(states)[-2]
        max_num_edges = tf.shape(graph.edges)[-2]
        with tf.control_dependencies([
            tf.assert_equal(max_num_nodes, graph.max_num_nodes),
            tf.assert_equal(max_num_edges, graph.max_num_edges),
            tf.assert_equal(tf.shape(states)[-3], graph.batch_size)
        ]):
            batch_size = graph.batch_size
            unknown_prefix = tf.shape(states)[:-3]
            static_unknown_prefix = states.shape.as_list()[:-3]
            full_prefix = tf.shape(states)[:-1]
            static_full_prefix = states.shape.as_list()[:-1]

        # (...,B,N,nh,dk) -> (B,N,...,nh,dk)
        perm = tf.range(states.shape.ndims + 1)
        # (...,-4,-3,-2,-1) -> (-4,-3,...,-2,-1)
        perm = tf.stack([
            perm[-4], perm[-3], *tf.unstack(perm[:-4]), perm[-2], perm[-1]
        ])
        as_queries = tf.transpose(DOT(states, query_transform), perm)
        as_keys = tf.transpose(DOT(states, key_transform), perm)

        # (B,N,...,nh,dk) --batch_gather[(B,E)]--> (B,E,...,nh,dk)
        sids, rids = graph.edges[..., 0], graph.edges[..., 1]
        sender_keys = tf.batch_gather(as_keys, indices=sids)
        receiver_queries = tf.batch_gather(as_queries, indices=rids)

        # (B,E,...,nh,dk) * (B,E,...,nh,dk) -> (B,E,...,nh)
        compatibilities = tf.math.reduce_sum(
            tf.math.multiply(receiver_queries, sender_keys), axis=-1
        )
        compatibilities = tf.math.divide(
            compatibilities, tf.math.sqrt(tf.cast(dim_key, tf.float32))
        )

        # (B,E,de) * (de,nh) -> (B,E,nh) -> (B,E,...,nh)
        edge_attr_bias = 0.0
        assert (graph.edge_attrs is not None) == has_edge_attr
        if has_edge_attr:
            edge_attr_bias = DOT(graph.edge_attrs, edge_attr_transform)
            for _ in range(len(static_unknown_prefix)):
                edge_attr_bias = tf.expand_dims(edge_attr_bias, axis=-2)

        compatibilities = tf.math.add(compatibilities, edge_attr_bias)

        # (B,E,...,nh) -> (...,nh,B,E)
        perm = tf.range(compatibilities.shape.ndims)
        perm = tf.stack([
            *tf.unstack(perm[2:-1]), perm[-1], perm[0], perm[1]
        ])
        compatibilities = tf.transpose(compatibilities, perm)
        reverse_mask = None if reverse_mask is None \
            else tf.math.logical_and(
                tf.fill(tf.shape(compatibilities), tf.constant(True)),
                tf.expand_dims(reverse_mask, axis=-3)
            )  # ([...,]B,E) -> ([...,]1,B,E) -> (...,nh,B,E)

        # (...,nh,B,E) -> (...*nh,B,E)
        flat_shape = tf.stack([
            # It is OK when `prefix_shape` is `[]`, because `tf.reduce_prod`
            # (Equivalent to `np.prod`) returns 1 for `[]`.
            tf.math.multiply(tf.math.reduce_prod(unknown_prefix), num_heads),
            batch_size, max_num_edges
        ])
        flat_compatibilities = tf.reshape(compatibilities, flat_shape)
        reverse_mask = None if reverse_mask is None \
            else tf.reshape(reverse_mask, flat_shape)

        flat_attention_weights = ATTENTION(
            graph=graph, similarities=flat_compatibilities,
            reverse_mask=reverse_mask
        )
        if reverse_mask is not None:
            flat_attention_weights = tf.math.multiply(
                flat_attention_weights,
                tf.math.subtract(1.0, util.float(reverse_mask))
            )

        # (...*nh,B,E) -> (...,nh,B,E,1)
        attention_weights_shape = tf.stack([
            *tf.unstack(unknown_prefix), num_heads, batch_size, max_num_edges
        ])
        attention_weights = tf.expand_dims(
            tf.reshape(flat_attention_weights, attention_weights_shape),
            axis=-1)
        attention_weights.set_shape(
            [*static_unknown_prefix, num_heads, None, None, 1]
        )

        # (...,B,N,d) -> (...,nh,B,E,dv)
        messages = MESSAGE(
            graph=graph, senders=states, receivers=states,
            global_states=global_states
        )

        # (...,nh,B,E,1) * (...,nh,B,E,dv) -> (...,nh,B,E,dv)
        weighted_messages = tf.math.multiply(messages, attention_weights)

        # (...,nh,B,E,dv) -> (B,E,...,nh,dv)
        perm = tf.range(weighted_messages.shape.ndims)
        # (...,-4,-3,-2,-1) -> (-3,-2,...,-4,-1)
        perm = tf.stack([
            perm[-3], perm[-2], *tf.unstack(perm[:-4]), perm[-4], perm[-1]
        ])
        per_edge_messages = tf.transpose(weighted_messages, perm)

        # (B) -> (B, 1, 1) -> (B, E, 1) -> (B, E, 2)
        expand = functools.partial(tf.expand_dims, axis=-1)
        recv_ids = expand(rids)
        batch_ids = tf.math.add(
            expand(expand(tf.range(batch_size, dtype=tf.int32))),
            tf.zeros_like(recv_ids)
        )
        scatter_indices_2d = tf.concat([batch_ids, recv_ids], axis=-1)

        # (B,E,...,nh,dv) --scatter[(B,E,2)]--> (B,N,...,nh,dv)
        aggregated_messages_shape = tf.stack([
            batch_size, max_num_nodes,
            *tf.unstack(unknown_prefix), num_heads, dim_value
        ])
        aggregated_messages = tf.manip.scatter_nd(
            indices=scatter_indices_2d,
            updates=per_edge_messages,
            shape=aggregated_messages_shape
        )
        aggregated_messages.set_shape(
            [None, None, *static_unknown_prefix, num_heads, dim_value]
        )

        # (B,N,...,nh,dv) -> (...,B,N,nh,dv) ->
        #     (...*B*N,nh,dv) * (nh,dv,d) -> (...*B*N,nh,d)
        perm = tf.range(aggregated_messages.shape.ndims)
        # (0,1,...,-2,-1) -> (...,0,1,-2,-1)
        perm = tf.stack([
            *tf.unstack(perm[2:-2]), perm[0], perm[1], perm[-2], perm[-1]
        ])
        flat_messages_shape = tf.stack([
            tf.math.reduce_prod(full_prefix), num_heads, dim_value
        ])
        flat_updates = tf.linalg.einsum(
            "ijk,jkl->ijl",
            tf.reshape(
                tf.transpose(aggregated_messages, perm),
                flat_messages_shape
            ),
            inverse_transform
        )

        # (...*B*N,nh,d) -> (...,B,N,nh,d)
        per_head_updates_shape = tf.stack([
            *tf.unstack(tf.shape(states)[:-1]), num_heads, dim_output
        ])
        per_head_updates = tf.reshape(flat_updates, per_head_updates_shape)
        per_head_updates.set_shape(
            [*static_unknown_prefix, None, None, num_heads, dim_output]
        )

        # (...,B,N,d) * (d,nh) -> (...,B,N,nh) -> (...,B,N,nh,1)
        per_head_weights = tf.math.softmax(tf.math.divide(
            DOT(states, head_transform),
            tf.math.sqrt(tf.cast(dim_input, tf.float32))
        ), axis=-1)
        per_head_weights = tf.expand_dims(per_head_weights, axis=-1)

        # (...,B,N,nh,1) * (...,B,N,nh,d) -> (...,B,N,nh,d) -> (...,B,N,d)
        updates = tf.math.reduce_sum(
            tf.math.multiply(per_head_weights, per_head_updates), axis=-2
        )
        updates.set_shape([*static_full_prefix, dim_output])

        return ACTIVATION(LAYER_NORM_O(SKIP_CONN(original_states, updates)))

    return call


def DenseGraphNN(config, name="DenseGraphNN"):
    num_heads = config.num_heads
    dim_input = config.dim_input
    dim_key = config.dim_key
    dim_value = config.dim_value
    dim_node_attr = config.dim_node_attr
    dim_edge_attr = config.dim_edge_attr
    dim_output = dim_input

    has_node_attr = (dim_node_attr > 0)
    has_edge_attr = (dim_edge_attr > 0)

    dim_input += (dim_node_attr + config.dim_global_state)

    ATTENTION = _lookup_attention_method(config)
    ACTIVATION = lookup_activation_fn(config.activation)
    SKIP_CONN = _make_skip_conn(config)

    initializer = tf.initializers.glorot_normal

    with tf.variable_scope(name):
        key_transform = tf.get_variable(
            "key_transform", shape=[dim_input, num_heads, dim_key],
            initializer=initializer(), trainable=True)
        query_transform = tf.get_variable(
            "query_transform", shape=[dim_input, num_heads, dim_key],
            initializer=initializer(), trainable=True)
        edge_attr_transform = tf.get_variable(
            "edge_attr_transform", shape=[dim_edge_attr, num_heads],
            initializer=initializer(), trainable=True)
        inverse_transform = tf.get_variable(
            "inverse_transform", shape=[num_heads, dim_value, dim_output],
            initializer=initializer(), trainable=True)
        head_transform = tf.get_variable(
            "head_transform", shape=[dim_input, num_heads],
            initializer=initializer(), trainable=True)

        MESSAGE = _make_message_fn(config)
        LAYER_NORM_I, LAYER_NORM_O = _make_layer_norm_fn(config)

    def aggregate_unary_messages(graph, global_states, states, weights):
        '''
        Args:
          states: A (..., B, N, d) Tensor.
          weights: A (..., nh, B, N, N) Tensor.

        Returns:
          aggregated_messages: A (..., B, N, nh, dv) Tensor.
        '''
        # (...,nh,B,N,dv)
        messages = MESSAGE(
            graph=graph, senders=states, receivers=states,
            global_states=global_states
        )

        # (...,nh,B,N,N)*(...,nh,B,N,dv) -> (...,nh,B,N,dv) -> (...,B,N,nh,dv)
        perm = tf.range(messages.shape.ndims)
        # (...,-4,-3,-2,-1) -> (...,-3,-2,-4,-1)
        perm = tf.stack([
            *tf.unstack(perm[:-4]), perm[-3], perm[-2], perm[-4], perm[-1]
        ])
        return tf.transpose(tf.linalg.matmul(weights, messages), perm)

    def aggregate_binary_messages(graph, global_states, states, weights):
        '''
        Args:
          states: A (..., B, N, d) Tensor.
          weights: A (..., nh, B, N[r], N[s]) Tensor.

        Returns:
          agg_msgs: A (..., B, N, nh, dv) Tensor.
        '''
        # (...,B,N,d) -> (...,B,N[r],N[s],nh,dv)
        pairwise_messages = MESSAGE(
            graph=graph, senders=states, receivers=states,
            global_states=global_states
        )

        # (...,nh,B,Nr,Ns) -> (...,B,Nr,Ns,nh,1) * (...,B,Nr,Ns,nh,dv)
        perm = tf.range(weights.shape.ndims)
        # (...,-4,-3,-2,-1) -> (...,-3,-2,-1,-4)
        perm = tf.stack([
            *tf.unstack(perm[:-4]), perm[-3], perm[-2], perm[-1], perm[-4]
        ])
        weights = tf.expand_dims(tf.transpose(weights, perm), axis=-1)
        weighted_messages = tf.math.multiply(pairwise_messages, weights)

        # (...,B,N[r],N[s],nh,dv) -> (...,B,N[r],nh,dv)
        aggregated_messages = tf.math.reduce_sum(weighted_messages, axis=-3)
        return aggregated_messages

    def call(graph, states, global_states=None, reverse_mask=None):
        '''
        Args:
          graph: A RuntimeGraph object.
          states: A (..., B, N, d) Tensor.
          global_states: Optional. A (..., B, d) Tensor.
          reverse_mask: Optional. A ([..., ]B, N, N) boolean Tensor.
        '''
        full_prefix = tf.shape(states)[:-1]
        static_full_prefix = states.shape.as_list()[:-1]
        static_unknown_prefix = states.shape.as_list()[:-3]

        # (B,N,1) * (...,B,N,d)
        original_states = states
        states = LAYER_NORM_I(states)
        states = graph.mask_nodal_info(states)
        states = _concat_node_attrs(graph, states, has_node_attr)
        states = _concat_global_states(states, global_states)

        # (...,B,N,d) * (d,nh,dk) -> (...,B,N,nh,dk) -> (...,nh,B,N,dk)
        perm = tf.range(states.shape.ndims + 1)
        # (...,-4,-3,-2,-1) -> (...,-2,-4,-3,-1)
        perm = tf.stack([
            *tf.unstack(perm[:-4]), perm[-2], perm[-4], perm[-3], perm[-1]
        ])
        queries = tf.transpose(DOT(states, query_transform), perm)
        keys = tf.transpose(DOT(states, key_transform), perm)

        # (...,nh,B,N,dk) * (...,nh,B,[N,dk].T) -> (...,nh,B,N,N)
        compatibilities = tf.linalg.matmul(keys, queries, transpose_b=True)
        compatibilities = tf.math.divide(
            compatibilities, tf.sqrt(tf.cast(dim_key, tf.float32))
        )

        # (B,N,N,de) * (de,nh) -> (B,N,N,nh) -> (nh,B,N,N) -> (...,nh,B,N,N)
        edge_attr_bias = 0.0
        assert (graph.dense_edge_attrs is not None) == has_edge_attr
        if has_edge_attr:
            edge_attr_bias = DOT(graph.dense_edge_attrs, edge_attr_transform)
            edge_attr_bias = tf.transpose(edge_attr_bias, [3, 0, 1, 2])
            for _ in range(len(static_unknown_prefix)):
                edge_attr_bias = tf.expand_dims(edge_attr_bias, axis=0)

        # (...,nh,B,N[s],N[r])
        compatibilities = tf.math.add(compatibilities, edge_attr_bias)
        if reverse_mask is not None:
            # ([...,]1,B,N[s],N[r]) -> (...,nh,B,N[s],N[r])
            reverse_mask = tf.math.logical_and(
                tf.expand_dims(reverse_mask, axis=-4),
                tf.fill(tf.shape(compatibilities), tf.constant(True))
            )

        # (...,nh,B,N[r],N[s])
        attention_weights = ATTENTION(
            graph=graph, similarities=compatibilities,
            reverse_mask=reverse_mask
        )
        if reverse_mask is not None:
            attention_weights = tf.math.multiply(
                attention_weights, tf.math.subtract(
                    1.0, util.float(tf.linalg.transpose(reverse_mask))
                )
            )  # (...,nh,B,N[r],N[s]) * (...,nh,B,{N[s],N[r]}.T)

        # (...,B,N,nh,dv)
        aggregate = aggregate_binary_messages \
            if config.messenger == MESSENGER_BINARY \
            else aggregate_unary_messages
        messages = aggregate(
            graph=graph, global_states=global_states, states=states,
            weights=attention_weights
        )

        # (...,B,N,nh,dv) -> (...*B*N,nh,dv)
        messages = tf.reshape(
            messages,
            [tf.math.reduce_prod(full_prefix), num_heads, dim_value]
        )
        # (...*B*N,nh,dv) * (nh,dv,d) -> (...*B*N,nh,d) -> (...,B,N,nh,d)
        per_head_updates = tf.reshape(
            tf.linalg.einsum("ijk,jkl->ijl", messages, inverse_transform),
            tf.stack([*tf.unstack(full_prefix), num_heads, dim_output])
        )
        per_head_updates.set_shape(
            [*static_full_prefix, num_heads, dim_output]
        )

        # (...,B,N,d) * (d,nh) -> (...,B,N,nh) -> (...,B,N,nh,1)
        per_head_weights = tf.math.softmax(tf.math.divide(
            DOT(states, head_transform),
            tf.math.sqrt(tf.cast(dim_input, tf.float32))
        ), axis=-1)
        per_head_weights = tf.expand_dims(per_head_weights, axis=-1)

        # (...,B,N,nh,1) * (...,B,N,nh,d) -> (...,B,N,nh,d) -> (...,B,N,d)
        updates = tf.math.reduce_sum(
            tf.math.multiply(per_head_weights, per_head_updates), axis=-2
        )
        updates.set_shape([*static_full_prefix, dim_output])

        return ACTIVATION(LAYER_NORM_O(SKIP_CONN(original_states, updates)))

    return call


def combine_lstm(combiner, states, updates, global_states=None):
    assert (
        (type(combiner) is keras.layers.StackedRNNCells) or
        (type(combiner) is keras.layers.LSTMCell)
    )
    lstm_states = util.pack_rnn_states(combiner, states)
    updates = _concat_global_states(updates, global_states)
    _, new_lstm_states = combiner(updates, lstm_states)
    return util.concat_rnn_states(new_lstm_states)


def combine_gru(combiner, states, updates, global_states=None):
    assert type(combiner) is keras.layers.GRUCell  # TODO
    updates = _concat_global_states(updates, global_states)
    new_states, _ = combiner(updates, [states])
    return new_states


def combine_add(combiner, states, updates, global_states=None):
    if global_states is not None:
        tf.logging.warning("Global states are not used in adder combiner.")
    del combiner, global_states
    return tf.math.add(states, updates)


_combine_methods = {
    COMBINER_LSTM: combine_lstm,
    COMBINER_GRU: combine_gru,
    COMBINER_ADD: combine_add
}


def readout_mean_max(gnn_config, graph, states):
    '''
    Args:
      states: A (..., N, dH) Tensor.

    Returns:
      summaries: A (..., 2*dh) Tensor.
    '''
    states = util.extract_rnn_output(
        states, gnn_config.rnn_num_layers, gnn_config.combiner
    )
    return tf.concat([
        tf.math.reduce_mean(states, axis=-2),
        tf.math.reduce_logsumexp(states, axis=-2)
    ], axis=-1)


_readout_methods = {
    READOUT_MEAN_MAX: readout_mean_max,
    READOUT_DISABLE: None
}


def StackedGraphNN(gnn_config, name="StackedGraphNN"):
    num_layers = gnn_config.num_layers
    share_gnn = gnn_config.recurrent
    combiner_type = gnn_config.combiner
    rnn_num_layers = gnn_config.rnn_num_layers
    dim_input = gnn_config.dim_input
    dim_update = dim_input + gnn_config.dim_global_state
    dim_full_state = gnn_config.dim_full_state
    dim_readout = dim_input * 2  # TODO

    READOUT = _readout_methods[gnn_config.readout]
    if READOUT is not None:
        READOUT = functools.partial(READOUT, gnn_config)

    with tf.variable_scope(name):
        graph_nn_list = []
        if num_layers <= 0:
            graph_nn_list = None
        elif share_gnn:
            graph_nn = GraphNN(gnn_config, name="GraphNN")
            graph_nn_list = [graph_nn] * num_layers
        else:
            for i in range(num_layers):
                graph_nn = GraphNN(gnn_config, name="GraphNN_{}".format(i))
                graph_nn_list.append(graph_nn)

        if READOUT is not None:
            global_gated_unit = util.gated_unit(
                dim_i=dim_readout, dim_o=dim_input,
                name="global_gated_unit"
            )

        if combiner_type == COMBINER_LSTM:
            combiner = util.make_lstm_cells(
                num_layers=rnn_num_layers,
                dim_in=dim_update,
                cell_size=gnn_config.dim_input,
                name="LSTMCombiner"
            )
        elif combiner_type == COMBINER_GRU:
            assert rnn_num_layers == 1  # TODO
            combiner = keras.layers.GRUCell(
                gnn_config.dim_input,
                input_shape=(dim_update,), name="GRUCell"
            )
        elif combiner_type == COMBINER_ADD:
            combiner = None
        else:
            raise ValueError("unknown combiner: " + combiner)

    COMBINE = functools.partial(
        _combine_methods[combiner_type], combiner=combiner)

    def _refresh_global(graph, global_states, states):
        if READOUT is None:
            return global_states
        elif global_states is None:
            return global_gated_unit(READOUT(graph, states))
        return tf.concat([
            global_states, global_gated_unit(READOUT(graph, states))
        ], axis=-1)

    def call(graph, states, global_states=None, reverse_mask=None):
        assert util.dim(states) == dim_full_state
        original_global_states = global_states

        for i in range(num_layers):
            global_states = _refresh_global(
                graph, original_global_states, states)
            gnn_inputs = util.extract_rnn_output(
                states, rnn_num_layers, combiner_type)
            updates = graph_nn_list[i](
                graph=graph, states=gnn_inputs,
                global_states=global_states, reverse_mask=reverse_mask
            )
            states = COMBINE(
                states=states, updates=updates,
                global_states=global_states
            )
        return states

    return call


def RecurrentGraphNN(gnn_config, dim_input,
                     rnn="LSTM", name="RecurrentGraphNN"):
    with tf.variable_scope(name):
        graph_nn = StackedGraphNN(gnn_config, name="StackedGraphNN")
        cell_size = gnn_config.dim_input

        if rnn == "LSTM":
            assert gnn_config.combiner == COMBINER_LSTM
            rnn_cell = util.make_lstm_cells(
                num_layers=gnn_config.rnn_num_layers,
                dim_in=dim_input, cell_size=cell_size
            )
        elif rnn == "GRU":
            assert gnn_config.combiner == COMBINER_GRU
            assert gnn_config.rnn_num_layers == 1  # TODO
            rnn_cell = keras.layers.GRUCell(
                cell_size, input_shape=(dim_input,),
                name="GRUCell"
            )
        else:
            raise ValueError("unknown rnn: " + rnn)

    def call(graph, initial_states, input_sequence):
        '''
        Args:
          init_states: A (..., N, d) Tensor.
          input_sequence: A (T, ..., N, dx) Tensor.

        Returns:
          final_states: A (..., N, d) Tensor.
          output_sequence: A (T, ..., N, d) Tensor.
        '''
        with tf.control_dependencies([tf.assert_equal(
            tf.shape(initial_states)[:-1], tf.shape(input_sequence)[1:-1]
        )]):
            len_input = tf.shape(input_sequence)[0]

        def cond(t, *unused_args):
            return tf.less(t, len_input)

        def body(t, rnn_states, output_sequence):
            _, new_rnn_states = rnn_cell(input_sequence[t], rnn_states)
            concat_states = util.concat_rnn_states(new_rnn_states)
            coupled_states = graph_nn(graph=graph, states=concat_states)
            new_output_sequence = output_sequence.write(t, coupled_states)
            new_rnn_states = util.pack_rnn_states(rnn_cell, coupled_states)
            return t + 1, new_rnn_states, new_output_sequence

        t0 = tf.constant(0)
        output_sequence = tf.TensorArray(tf.float32, size=len_input)
        initial_rnn_states = util.pack_rnn_states(rnn_cell, initial_states)

        _, last_rnn_states, output_sequence = tf.while_loop(
            cond, body,
            [t0, initial_rnn_states, output_sequence]
        )
        output_sequence = output_sequence.stack()

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(output_sequence)[0], len_input)
        ]):
            output_sequence.set_shape([
                *input_sequence.shape.as_list()[:-1], None
            ])
            last_states = util.concat_rnn_states(last_rnn_states)

        return last_states, output_sequence

    return call


############################################
#               Deprecated.                #
############################################


def SkipConnectedGraphNN(gnn_config, num_layers, dim_input,
                         name="SkipConnectedGraphNN"):
    with tf.variable_scope(name):
        graph_nn_list = []
        for i in range(num_layers):
            graph_nn = GraphNN(gnn_config, name="GraphNN_{}".format(i))
            graph_nn_list.append(graph_nn)

        lstm_cell = keras.layers.LSTMCell(
            gnn_config.dim_input, input_shape=(dim_input,), name="LSTMCell"
        )
        layer_norm = layer_norm_1d(gnn_config.dim_input, name="LayerNorm")

    def call(graph, init_node_states, inputs):
        '''
        Args:
          init_node_states: A (..., N, d) Tensor.
          inputs: A ([..., ]N, dx) Tensor.

        Returns:
          final_node_states: A (..., N, d) Tensor.
        '''
        init_cell_states = tf.zeros_like(init_node_states)

        node_states, cell_states = init_node_states, init_cell_states
        for i in range(num_layers + 1):
            _, new_states = lstm_cell(
                inputs, [node_states, cell_states]
            )
            if i >= num_layers:
                new_node_states = new_states[0]
            else:
                new_node_states = graph_nn_list[i](
                    graph, layer_norm(new_states[0])
                )
            new_cell_states = new_states[1]
            node_states, cell_states = new_node_states, new_cell_states
        final_cell_states = cell_states

        return layer_norm(final_cell_states)

    return call
