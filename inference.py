from __future__ import absolute_import
from __future__ import print_function

from gnn import StackedGraphNN
from util import mlp_mix_diag_normal

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

tfd = tfp.distributions

############################################
#               Deprecated.                #
############################################


class InferenceModel:
    def __init__(
        self, gen_model,
        dim_observ, dim_latent, dim_hidden,
        dim_mlp, gnn_num_layers, gnn_config,
        name="inference_model"
    ):
        gnn_config = gnn_config.clone()
        gnn_config.dim_input = dim_latent

        with tf.variable_scope(name):
            self._gnn_backward = StackedGraphNN(
                gnn_config, gnn_num_layers,
                name="gnn_backward")
            self._rnncell = keras.layers.GRUCell(
                dim_latent, input_shape=(dim_observ,),
                name="rnn_cell")
            self._gnn_forward = StackedGraphNN(
                gnn_config, gnn_num_layers,
                name="gnn_forward")
            self._mlp_normal_initial = mlp_mix_diag_normal(
                dim_latent, dim_mlp, dim_latent,
                name="mlp_posterior_initial")
            self._mlp_normal = mlp_mix_diag_normal(
                dim_latent, dim_mlp, dim_latent,
                name="mlp_posterior")
            self._history_to_hidden = keras.layers.Dense(
                dim_latent, input_shape=(dim_hidden,),
                name="history_to_hidden")
            self._future_to_scale = keras.layers.Dense(
                dim_latent, input_shape=(dim_latent,),
                name="future_to_scale")
            self._future_to_shift = keras.layers.Dense(
                dim_latent, input_shape=(dim_latent,),
                name="future_to_shift")

        self._dim_latent = dim_latent
        self._gen_model = gen_model

    def summarize_backward(self, graph, observs):
        '''
        Summarize the observations backward.

        Args:
          observs: A (T, N, dx) Tensor.
          graph: A RuntimeGraph object.

        Returns:
          summaries: A (T, N, dz) Tensor.
        '''
        shape = tf.shape(observs)
        num_steps, num_nodes = shape[0], shape[1]
        observs = tf.reverse(observs, [0])
        init_state = tf.zeros([num_nodes, self._dim_latent])

        def cond(t, *unused_args):
            return tf.less(t, num_steps)

        def body(t, state, summaries):
            new_state, _ = self._rnncell(observs[t], [state])
            evolved_state = self._gnn_backward(graph.reverse(), new_state)
            new_summaries = summaries.write(t, evolved_state)
            t = tf.add(t, 1)
            return t, evolved_state, new_summaries

        t = tf.constant(0)
        summaries = tf.TensorArray(tf.float32, size=num_steps)

        _, _, summaries = tf.while_loop(
            cond, body, [t, init_state, summaries]
        )

        summaries = summaries.stack()
        summaries = tf.reverse(summaries, [0])
        return summaries

    def init(self, reversed_graph, observations, lookaheads):
        '''
        Args:
          lookaheads: A (N, dz) Tensor.
          reversed_graph: A RuntimeGraph object.

        Returns:
          A distribution with batch shape (N) and event shape (dz)
        '''
        summaries = self._gnn_backward(reversed_graph, lookaheads)
        return self._mlp_normal_initial(summaries)

    def _forward_combine(self, graph, history_summaries, lookaheads):
        # future_summary = tf.reshape(
        #     tf.tile(
        #         future_summary,
        #         [tf.reduce_prod(tf.shape(prev_state)[:-2]), 1]),
        #     tf.shape(prev_state)
        # )
        # combined = tf.concat([new_state, future_summary], -1)

        # Above concatenation-based method suffers from the "latent variable
        # collapse" problem: q(z|x) = p(z), i.e., the inference network
        # may ignore conditional inputs. Here we use "conditional affine
        # transformation"[1] to force the inference network into depending
        # on the summaries of future observations. See:
        # [1] "Feature-wise transformations".
        # --> https://distill.pub/2018/feature-wise-transformations/
        hidden = self._history_to_hidden(history_summaries)
        scale = self._future_to_scale(lookaheads)
        shift = self._future_to_shift(lookaheads)
        combined = tf.math.add(tf.multiply(hidden, scale), shift)
        evolved = self._gnn_forward(graph, combined)
        return evolved

    def propose(self, graph, states, histories, observations, lookaheads):
        '''
        Args:
          states: A (S, N, dz) Tensor.
          lookaheads: A (N, dz) Tensor.

        Returns:
          A distribution with batch shape (S, N) and event shape (dz)
        '''
        model = self._gen_model
        new_histories = model.refresh_histories(histories, states)
        combined = self._forward_combine(graph, new_histories, lookaheads)
        return self._mlp_normal(combined)

    def posterior_srnn(self, graph, prev_states, prior_dists, lookaheads):
        '''
        Args:
          prev_state: A (S, N, dz) Tensor.
          prior_dist: Distribution with batch shape (S) and event shape (N, dz)
          future_summary: A (N, dz) Tensor.
          graph: A RuntimeGraph object.

        Returns:
          A distribution with batch shape (S, N) and event shape (dz)
        '''
        prior_locs = prior_dists.mean()
        # avg_prior_locs = tf.reduce_mean(prior_locs, 0)
        combined = self._forward_combine(graph, prev_states, lookaheads)
        distributions = self._mlp_normal(combined)
        locs, scales = distributions.mean(), distributions.stddev()
        locs = tf.math.add(locs, prior_locs)
        return tfd.MultivariateNormalDiag(locs, scales)
