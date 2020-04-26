from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from gnn import StackedGraphNN, GraphNN
import gnn

from util import mlp_diag_normal
from tgraph import tf_pad_axis_to
import util

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def compute_aux_scores(aux_model, graph, histories, states,
                       observations, beliefs, lookaheads):
    '''
    Returns:
      scores: A (..., B) Tensor.
    '''
    if aux_model is None:
        global_states, _ = states
        return tf.zeros(tf.shape(global_states)[:-1])

    if not (type(aux_model) is ZForcing or type(aux_model) is MIX):
        observations = observations[0]

    scores = aux_model.score(
        graph, histories, states,
        observations, beliefs, lookaheads
    )
    return graph.reduce_sum_nodal(scores)  # (..., B, N) -> (..., B)


def _broadcast_and_concat(global_states, local_states):
    global_states = tf.expand_dims(global_states, axis=-2)
    return util.broadcast_concat(local_states, global_states)


class ZForcing(object):
    def __init__(self, dim_latent, dim_summary, dim_observ,
                 dim_mlp=128, mlp_num_layers=1, mix_num_components=1,
                 num_future_steps=5, name="ZForcing"):
        with tf.variable_scope(name):
            self._mlp_diag_normal = mlp_diag_normal(
                dim_in=(2 * dim_latent), dim_hid=dim_mlp,
                dim_out=(num_future_steps * dim_observ),
                mlp_num_layers=mlp_num_layers
            )
        self._num_future_steps = num_future_steps

    def score(self, graph, histories, states,
              observations, beliefs, lookaheads):
        '''
        Args:
          histories: A 2-ary tuple.
          states: A 2-ary tuple.
          observations: A (F, B, N, dx) Tensor.

        Returns:
          scores: A (..., B, N) Tensor.
        '''
        del graph, histories, beliefs, lookaheads
        global_states, local_states = states
        combined_states = _broadcast_and_concat(*states)

        assert observations.shape.ndims == 4
        num_future_steps = self._num_future_steps
        num_remain_steps = tf.shape(observations)[0]
        dim_observ = util.dim(observations)

        mask = tf.cond(
            tf.math.greater_equal(num_remain_steps, num_future_steps),
            lambda: tf.ones(num_future_steps),
            lambda: tf.concat([
                tf.ones(num_remain_steps),
                tf.zeros(num_future_steps - num_remain_steps)
            ], axis=0)
        )
        observations = tf.cond(
            tf.math.greater_equal(tf.shape(observations)[0], num_future_steps),
            lambda: observations[:num_future_steps],
            lambda: tf_pad_axis_to(observations, axis=0, size=num_future_steps)
        )

        # (F, B, N, dx) -> (B, N, F, dx)
        perm = [1, 2, 0, 3]
        transposed_observations = tf.transpose(observations, perm)

        cond_dist = self._mlp_diag_normal(combined_states).build()
        assert type(cond_dist) is tfd.MultivariateNormalDiag
        # (..., B, N, F * dx)
        locs, scales = cond_dist.mean(), cond_dist.stddev()
        # (..., B, N, F, dx)
        shape = tf.stack([
            *tf.unstack(tf.shape(local_states)[:-1]),
            num_future_steps, dim_observ
        ])
        locs, scales = tf.reshape(locs, shape), tf.reshape(scales, shape)

        cond_dist = tfd.MultivariateNormalDiag(loc=locs, scale_diag=scales)
        log_probs = cond_dist.log_prob(transposed_observations)  # (B, N, F)
        return tf.math.reduce_sum(tf.math.multiply(log_probs, mask), axis=-1)


class CPC(object):
    def __init__(self, dim_latent, dim_summary, state="z", name="CPC"):
        if state == "z":
            dim_state = dim_latent * 2
        elif state == "h":
            dim_state = dim_summary * 2
        else:
            raise ValueError("Invalid argument: {}".format(state))

        with tf.variable_scope(name):
            self._linear = tf.get_variable(
                "linear", shape=[dim_state, dim_summary],
                trainable=True, initializer=tf.initializers.orthogonal()
            )
        self._dim_summary = dim_summary
        self._state = state

    def score(self, graph, histories, states,
              observations, beliefs, lookaheads):
        '''
        Args:
          histories: A 2-ary tuple:
            - global_histories: A (..., B, dH) Tensor.
            - local_histories: A (..., B, N, dH) Tensor.
          states: A 2-ary tuple:
            - global_states: A (..., B, dz) Tensor.
            - local_states: A (..., B, N, dz) Tensor.
        '''
        del graph, beliefs
        global_histories, local_histories = histories
        global_states, local_states = states

        shape = tf.shape(lookaheads)
        with tf.control_dependencies([
            tf.assert_equal(tf.size(shape), 3),
            tf.assert_equal(shape[:-1], tf.shape(local_states)[-3:-1])
        ]):
            summaries = lookaheads
            batch_size, max_num_nodes = shape[0], shape[1]
            unknown_prefix = tf.shape(local_states)[:-3]
            unknown_prefix_list = tf.unstack(unknown_prefix)

        local_context = _broadcast_and_concat(global_states, local_states) \
            if self._state == "z" \
            else _broadcast_and_concat(global_histories, local_histories)

        # (..., B, N, dz) ->
        # (..., B * N, dz) * (dz, ds) -> (..., B * N, ds)
        flat_local_context = tf.reshape(local_context, tf.stack([
            *unknown_prefix_list, batch_size * max_num_nodes, -1
        ]))
        transformed_local_context = tf.linalg.tensordot(
            flat_local_context, self._linear, axes=1
        )

        # (B, N, ds) -> (B * N, ds) -> (..., B * N, ds)
        broadcast_summaries = tf.math.add(
            tf.zeros_like(transformed_local_context),
            tf.reshape(summaries, tf.stack([batch_size * max_num_nodes, -1]))
        )

        # (..., B * N, ds) * (..., [B * N, ds].T) -> (..., B * N, B * N)
        pairwise_log_bilinear_scores = tf.math.divide(
            tf.linalg.matmul(
                transformed_local_context,
                broadcast_summaries, transpose_b=True
            ),
            tf.math.sqrt(util.float(self._dim_summary))
        )
        # (..., B * N, B * N) -> (..., B * N) -> (..., B, N)
        batched_nce_scores = tf.math.subtract(
            tf.linalg.diag_part(pairwise_log_bilinear_scores),
            tf.math.reduce_logsumexp(pairwise_log_bilinear_scores, axis=-1)
        )
        return tf.reshape(batched_nce_scores, tf.stack([
            *unknown_prefix_list, batch_size, max_num_nodes
        ]))


class DGI(object):
    def __init__(self, dim_latent, dim_summary, name="DGI"):
        ''' Deep Graph Infomax '''
        with tf.variable_scope(name):
            self._linear = tf.get_variable(
                "linear", shape=[dim_summary, dim_summary],
                trainable=True, initializer=tf.initializers.orthogonal()
            )
            self._bilinear = tf.get_variable(
                "bilinear", shape=[dim_latent, dim_summary],
                trainable=True, initializer=tf.initializers.orthogonal()
            )
        self._dim_summary = dim_summary

    def score(self, graph, histories, states,
              observations, beliefs, lookaheads):
        '''
        Args:
          histories: A 2-ary tuple:
            - global_histories: A (..., B, dH) Tensor.
            - local_histories: A (..., B, N, dH) Tensor.
          states: A 2-ary tuple:
            - global_states: A (..., B, dz) Tensor.
            - local_states: A (..., B, N, dz) Tensor.
        '''
        del beliefs, lookaheads
        _, local_histories = histories
        global_states, _ = states

        # (..., B, N, dz) * (dz, ds) -> (..., B, N, ds)
        transformed_local_embeddings = tf.linalg.tensordot(
            local_histories, self._linear, axes=1
        )
        # (..., B, N, ds) -> (..., B, ds) * (ds, ds) -> (..., B, ds)
        transformed_graph_readouts = tf.linalg.tensordot(
            global_states, self._bilinear, axes=1
        )

        # (..., B, ds) -> (..., B, B, ds)
        shape_list = tf.unstack(tf.shape(transformed_graph_readouts))
        broadcast_graph_readouts = tf.math.add(
            tf.zeros(tf.stack([
                *shape_list[:-1], *shape_list[-2:]
            ])),
            tf.expand_dims(transformed_graph_readouts, axis=-3)
        )

        # (..., B, N, ds) * (..., B, [B, ds].T) -> (..., B, N, B)
        global_local_bilinear_scores = tf.linalg.matmul(
            transformed_local_embeddings,
            broadcast_graph_readouts, transpose_b=True
        )
        scaled_scores = tf.math.divide(
            global_local_bilinear_scores,
            tf.math.sqrt(util.float(self._dim_summary))
        )

        # (..., B1, N, B2) -> (..., N, B1, B2)
        perm = tf.range(scaled_scores.shape.ndims)
        # (..., -3, -2, -1) -> (..., -2, -3, -1)
        perm = tf.stack([
            *tf.unstack(perm[:-3]), perm[-2], perm[-3], perm[-1]
        ])
        transposed_scores = tf.transpose(scaled_scores, perm)

        # (..., N, B, B) -> (..., N, B) -> (..., B, N)
        batched_nce_scores = tf.math.subtract(
            tf.linalg.diag_part(transposed_scores),
            tf.math.reduce_logsumexp(transposed_scores, axis=-1)
        )
        return tf.linalg.transpose(batched_nce_scores)


class EdgeClassifier(object):
    def __init__(self, dim_latent, dim_summary, name="EdgeClassifier"):
        dim_concat = dim_summary + dim_latent
        with tf.variable_scope(name):
            self._linear = tf.get_variable(
                "linear", shape=[dim_concat, dim_concat],
                trainable=True, initializer=tf.initializers.orthogonal()
            )
        self._dim_concat = dim_concat

    def score(self, graph, histories, states,
              observations, beliefs, lookaheads):
        del beliefs, lookaheads
        _, local_histories = histories
        _, local_states = states
        batch_shape = tf.shape(local_states)[:-2]

        full_states = tf.concat([local_histories, local_states], axis=-1)
        pairwise_bilinear_scores = tf.linalg.matmul(
            tf.linalg.tensordot(full_states, self._linear, axes=1),
            full_states, transpose_b=True
        )
        scaled_pairwise_bilinear_scores = tf.math.divide(
            pairwise_bilinear_scores,
            tf.math.sqrt(util.float(self._dim_concat))
        )

        logits = tf.reshape(
            scaled_pairwise_bilinear_scores,
            shape=tf.stack([
                *tf.unstack(batch_shape), tf.math.square(graph.num_nodes)
            ])
        )
        labels = tf.reshape(tf.sparse.to_dense(graph.adjacency), shape=[-1])
        broadcast_labels = tf.math.add(
            util.float(labels), tf.zeros_like(logits)
        )

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=broadcast_labels, logits=logits
        )
        return tf.math.negative(tf.math.reduce_sum(cross_entropy, axis=-1))


class MaskedGNNDecoder(object):
    def __init__(self, dim_latent, dim_summary, gnn_config,
                 dim_mlp=None, all_at_once=False, num_masked_nodes=1,
                 name="MaskedGNNDecoder"):
        gnn_config.dim_input = dim_summary
        gnn_config.skip_conn = False
        gnn_config.layer_norm_out = False
        if all_at_once:
            gnn_config = gnn_config.single_layer
            gnn_config.num_heads = 1
            gnn_config.dim_value = gnn_config.dim_input
            gnn_config.attention = gnn.ATTENTION_UNIFORM
            gnn_config.messenger = gnn.MESSENGER_UNARY
            gnn_config.activation = "swish"
        else:
            gnn_config.combiner = gnn.COMBINER_ADD
            gnn_config.readout = gnn.READOUT_MEAN_MAX
            gnn_config.dim_global_state = gnn_config.dim_readout
        dim_mlp = dim_mlp or dim_summary

        with tf.variable_scope(name):
            if all_at_once:
                self._gnn = GraphNN(gnn_config, "gnn_one_layer")
            else:
                self._gnn = StackedGraphNN(gnn_config, "masked_gnn")

            self._mlp = util.mlp_two_layers(
                dim_in=dim_summary, dim_out=dim_summary,
                dim_hid=dim_mlp, act_out="tanh",
                name="mlp_two_layers"
            )
            self._linear = tf.get_variable(
                "linear", shape=[dim_summary, dim_summary],
                trainable=True, initializer=tf.initializers.orthogonal()
            )
        self._num_masked_nodes = num_masked_nodes
        self._all_at_once = all_at_once
        self._gnn_is_sparse = (gnn_config.impl == gnn.GNN_SPARSE)

    def _compute_nce_score(self, predictions, lookaheads):
        assert predictions.shape.ndims == lookaheads.shape.ndims
        shape = tf.shape(predictions)
        batch_size, max_num_nodes = shape[-3], shape[-2]
        unknown_prefix = shape[:-3]
        unknown_prefix_list = tf.unstack(unknown_prefix)
        # (..., B, N, dh) -> (..., B * N, dh)
        flat_shape = tf.stack([
            *unknown_prefix_list, batch_size * max_num_nodes, -1
        ])
        # * (dh, ds) -> (..., B * N, ds)
        flat_predictions = tf.reshape(predictions, flat_shape)
        flat_lookaheads = tf.reshape(lookaheads, flat_shape)

        # (..., B * N, ds) * (..., [B * N, ds].T) -> (..., B * N, B * N)
        pairwise_log_bilinear_scores = tf.math.divide(
            tf.linalg.matmul(
                flat_predictions, tf.linalg.tensordot(
                    flat_lookaheads, self._linear, axes=1
                ), transpose_b=True
            ),
            tf.math.sqrt(util.float(util.dim(predictions)))
        )
        # (..., B * N, B * N) -> (..., B * N) -> (..., B, N)
        batched_nce_scores = tf.math.subtract(
            tf.linalg.diag_part(pairwise_log_bilinear_scores),
            tf.math.reduce_logsumexp(pairwise_log_bilinear_scores, axis=-1)
        )
        return tf.reshape(batched_nce_scores, tf.stack([
            *unknown_prefix_list, batch_size, max_num_nodes
        ]))

    def _score_all_at_once(self, graph, local_histories, lookaheads):
        prefix_shape = tf.shape(local_histories)[-3:-1]  # (B, N)
        mask = graph.gen_sparse_self_loop_mask(prefix_shape) \
            if self._gnn_is_sparse \
            else graph.gen_dense_self_loop_mask(prefix_shape)
        hidden = self._gnn(
            graph=graph, states=local_histories,
            reverse_mask=mask
        )
        return self._compute_nce_score(self._mlp(hidden), lookaheads)

    def _score_recv_only(self, graph, local_histories, lookaheads):
        prefix_shape = tf.shape(local_histories)[-4:-1]  # (S, B, N)
        num_samples, batch_size, num_nodes = tf.unstack(prefix_shape)

        node_ids = tf.random.uniform(
            shape=[num_samples, batch_size, self._num_masked_nodes],
            dtype=tf.int32, minval=0, maxval=num_nodes
        )
        # (S, B, N, N) and (S, B, N)
        adj_mask, node_mask = graph.gen_dense_recv_only_mask(node_ids)
        mask = adj_mask if not self._gnn_is_sparse else \
            graph.gen_sparse_recv_only_mask(node_ids)  # (S, B, E)

        masked_local_histories = tf.math.multiply(
            local_histories, tf.math.subtract(
                1.0, tf.expand_dims(util.float(node_mask), axis=-1)
            )
        )
        hidden = self._gnn(
            graph=graph, states=masked_local_histories,
            reverse_mask=mask
        )
        scores = self._compute_nce_score(self._mlp(hidden), lookaheads)
        return tf.math.multiply(scores, util.float(node_mask))

    def score(self, graph, histories, states,
              observations, beliefs, lookaheads):
        '''
        Args:
          histories: A 2-ary tuple:
            - global_histories: A (..., S, B, dH) Tensor.
            - local_histories: A (..., S, B, N, dH) Tensor.
          states: A 2-ary tuple:
            - global_states: A (..., S, B, dz) Tensor.
            - local_states: A (..., S, B, N, dz) Tensor.
        '''
        del states, beliefs, observations
        _, local_histories = histories

        # (B, N, ds) -> (..., B, N, ds)
        lookaheads = tf.math.add(lookaheads, tf.zeros_like(local_histories))
        assert local_histories.shape.ndims >= 4
        assert local_histories.shape.ndims == lookaheads.shape.ndims

        fn = self._score_all_at_once \
            if self._all_at_once else self._score_recv_only
        return fn(graph, local_histories, lookaheads)


class MIX(object):
    def __init__(self, dim_latent, dim_summary,
                 dim_observ, gnn_config,
                 dim_mlp=128, mlp_num_layers=2,
                 cpc_scale=1.0, dgi_scale=1.0,
                 zf_scale=0.0, mask_scale=1.0,
                 cpc_state="z", zf_num_future_steps=5,
                 mask_all_at_once=False, mask_num_nodes=1,
                 name="MIX"):
        if cpc_scale != 0.0:
            self._cpc = CPC(dim_latent, dim_summary, state=cpc_state)
        if dgi_scale != 0.0:
            self._dgi = DGI(dim_latent, dim_summary)
        if zf_scale != 0.0:
            self._zf = ZForcing(
                dim_latent, dim_summary, dim_observ,
                dim_mlp=dim_mlp, mlp_num_layers=mlp_num_layers,
                num_future_steps=zf_num_future_steps
            )
        if mask_scale != 0.0:
            self._mask = MaskedGNNDecoder(
                dim_latent, dim_summary, gnn_config,
                dim_mlp=dim_mlp,
                all_at_once=mask_all_at_once,
                num_masked_nodes=mask_num_nodes
            )

        self._cpc_scale = cpc_scale
        self._dgi_scale = dgi_scale
        self._zf_scale = zf_scale
        self._mask_scale = mask_scale

    @property
    def num_future_steps(self):
        return self._zf._num_future_steps

    def score(self, graph, histories, states, observations, *args, **kwargs):
        _, local_states = states
        zero_scores = tf.zeros(tf.shape(local_states)[:-1])
        params = [graph, histories, states, observations[0], *args]
        cpc_scores = zero_scores if self._cpc_scale == 0.0 \
            else self._cpc.score(*params, **kwargs)
        dgi_scores = zero_scores if self._dgi_scale == 0.0 \
            else self._dgi.score(*params, **kwargs)
        mask_scores = zero_scores if self._mask_scale == 0.0 \
            else self._mask.score(*params, **kwargs)
        params[3] = observations
        zf_scores = zero_scores if self._zf_scale == 0.0 \
            else self._zf.score(*params, **kwargs)
        return tf.math.add_n([
            self._cpc_scale * cpc_scores,
            self._dgi_scale * dgi_scores,
            self._zf_scale * zf_scores,
            self._mask_scale * mask_scores
        ])
