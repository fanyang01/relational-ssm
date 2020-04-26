from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gnn import StackedGraphNN
from util import mlp_two_layers
from util import mlp_mix_diag_normal
from util import layer_norm_1d

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

tfd = tfp.distributions


def _init_histories_like(states, dim):
    shape = tf.stack([*tf.unstack(tf.shape(states)[:-1]), dim])
    histories = tf.zeros(shape)
    histories.set_shape([*states.shape[:-1].as_list(), dim])
    return histories


def make_gated_trans_dist(dim_in, dim_mlp,
                          loc_activation="tanh", scale_shift=1e-3,
                          name="gated_transition"):
    with tf.variable_scope(name):
        in_to_gate = mlp_two_layers(
            dim_in, dim_mlp, dim_in, nl_out="sigmoid",
            name="in_to_gate")
        in_to_hid = mlp_two_layers(
            dim_in, dim_mlp, dim_in, nl_out=loc_activation,
            name="in_to_hid")
        in_to_loc = keras.layers.Dense(
            dim_in, input_shape=(dim_in,), activation=loc_activation,
            kernel_initializer=keras.initializers.identity(),
            name="in_to_loc")
        hid_to_scale = keras.Sequential([
            keras.layers.Activation("softplus", input_shape=(dim_in,)),
            keras.layers.Dense(dim_in),
            keras.layers.Activation("softplus")
        ], name="hid_to_scale")

    def trans(z):
        gate = in_to_gate(z)
        hidden = in_to_hid(z)
        loc = tf.add(
            tf.multiply(gate, hidden),
            tf.multiply(tf.subtract(1.0, gate), in_to_loc(z))
        )
        scale = tf.add(hid_to_scale(hidden), scale_shift)
        return tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=scale, name="trans_dist"
        )
    return trans


class MarkovModel:
    '''
    Generative model for Markovian GNN-SSM:

    p(X_{0:T}, Z_{0:T}) = p(Z_0)p(X_0|Z_0)
                          \\prod _{t=1}^{T}{p(Z_t|Z_{t-1})p(X_t|Z_t)}
    '''

    def __init__(
        self,
        dim_observ, dim_latent, dim_hidden, dim_mlp, gnn_config,
        trans_mlp_num_layers=1, emit_mlp_num_layers=1,  # TODO
        trans_mix_num_components=1, emit_mix_num_components=1,  # TODO
        trans_activation="tanh", trans_scale_shift=1e-3,
        emit_activation="linear", emit_scale_shift=1e-3,
        emit_neg_binomial=False,  # TODO
        emit_identity=True,  # TODO
        init_mix_num_components=1,  # TODO
        name="markov_gen_model"
    ):
        gnn_config = gnn_config.clone()
        gnn_config.dim_input = dim_latent

        init_state_loc = tf.zeros(dim_latent)
        init_state_scale = tf.ones(dim_latent)

        with tf.variable_scope(name):
            self._gnn = StackedGraphNN(
                gnn_config, name="trans_gnn"
            )
            self._prior = tfd.MultivariateNormalDiag(
                loc=init_state_loc, scale_diag=init_state_scale, name="prior"
            )
            self._emit_dist = mlp_mix_diag_normal(
                dim_in=dim_latent, dim_hid=dim_mlp, dim_out=dim_observ,
                mix_num_components=emit_mix_num_components,
                mlp_num_layers=emit_mlp_num_layers,
                loc_activation=emit_activation,
                scale_shift=emit_scale_shift,
                name="emit_dist"
            )
            self._trans_dist = make_gated_trans_dist(
                dim_latent, dim_mlp,
                loc_activation=trans_activation,
                scale_shift=trans_scale_shift,
                name="trans_dist"
            )
            self._state_to_hidden = keras.layers.Dense(
                dim_hidden, input_shape=(dim_latent,),
                activation="tanh", name="state_to_hidden"
            )
            self._layer_norm = layer_norm_1d(
                dim_hidden, name="layer_norm_1d"
            )
        self._dim_latent = dim_latent
        self._dim_hidden = dim_hidden
        self._dim_observ = dim_observ

    @property
    def dim_observ(self):
        return self._dim_observ

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def dim_hidden(self):
        return self._dim_hidden

    '''
    Following three are dummy methods for imitating the behavior of
    non-Markov models.
    '''

    def init_histories_like(self, states):
        return _init_histories_like(states, self._dim_hidden)

    def refresh_histories(self, unused_histories, states):
        return self._state_to_hidden(states)

    '''
    The generative model.
    '''

    def prior(self):
        '''
        Returns:
          A distribution p(z_0) with event shape (dz).
        '''
        return self._prior

    def emit(self, histories, states):
        '''
        Z_t -> X_t

        Args:
          states: A (..., N, dz) Tensor.

        Returns:
          A distribution with batch shape (..., N) and event shape (dx).
        '''
        del histories
        return self._emit_dist(states)

    def trans(self, graph, histories, states):
        '''
        Z_t -> Z_{t+1}

        Args:
          states: A (..., N, dz) Tensor.

        Returns:
          A distribution with batch shape (..., N) and event shape (dz).
        '''
        new_histories = self.refresh_histories(histories, states)
        coupled_states = self._gnn(graph, self._layer_norm(states))
        trans_dist = self._trans_dist(coupled_states)
        return trans_dist, new_histories
