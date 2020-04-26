from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gnn import StackedGraphNN
from gnn import readout_mean_max
from flow import init_real_nvp, real_nvp_wrapper
from flow import init_perm_equiv_flow, perm_equiv_flow_wrapper
from util import mlp_mix_diag_normal, mlp_low_rank_normal
from util import mlp_mix_neg_binomial, get_mlp_mix_loc_scale_builder
from util import gated_unit, skip_cond_gated_unit, layer_norm_1d
from util import make_trainable_gmm, identity_diag_normal
import util

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


tfd = tfp.distributions


class External(object):
    def __init__(self, global_inputs, local_inputs):
        self._global_inputs = global_inputs
        self._local_inputs = local_inputs

    @property
    def global_inputs(self):
        return self._global_inputs

    @property
    def local_inputs(self):
        return self._local_inputs


class ExternalSequences(External):
    def __init__(self, global_inputs, local_inputs):
        super(ExternalSequences, self).__init__(
            global_inputs, local_inputs
        )

    def _assert_seq_len(self, sequence, length):
        if sequence is None or length is None:
            return sequence
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(sequence)[0], length)
        ]):
            return tf.identity(sequence)

    def _index_or_none(self, sequence, t, to=None):
        if sequence is None:
            return None
        if to is None:
            return sequence[t]
        return sequence[t:to]

    def current(self, t, to=None):
        global_inputs = self._index_or_none(self.global_inputs, t, to)
        local_inputs = self._index_or_none(self.local_inputs, t, to)
        if to is None:
            return External(global_inputs, local_inputs)
        return ExternalSequences(global_inputs, local_inputs)

    def truncate(self, length, total_length=None):
        global_inputs = self._assert_seq_len(self.global_inputs, total_length)
        local_inputs = self._assert_seq_len(self.local_inputs, total_length)
        return ExternalSequences(
            self._index_or_none(global_inputs, 0, length),
            self._index_or_none(local_inputs, 0, length)
        )


def _init_zeros(prefix_shape, dim):
    prefix_static_shape = prefix_shape
    if type(prefix_shape) is tf.TensorShape:
        prefix_static_shape = prefix_shape.as_list()
    elif type(prefix_shape) is not list:
        prefix_shape = tf.unstack(prefix_shape)
        prefix_static_shape = [None] * len(prefix_shape)
    shape = tf.stack([*prefix_shape, dim])
    zeros = tf.zeros(shape)
    zeros.set_shape([*prefix_static_shape, dim])
    return zeros


def refresh_histories_using_lstm(lstm_cell, histories, states):
    assert (
        (type(lstm_cell) is keras.layers.LSTMCell) or
        (type(lstm_cell) is keras.layers.StackedRNNCells)
    )
    lstm_states = util.pack_rnn_states(lstm_cell, histories)
    _, new_lstm_states = lstm_cell(inputs=states, states=lstm_states)
    new_histories = util.concat_rnn_states(new_lstm_states)
    return new_histories


class NonMarkovTransition(object):
    def __init__(self, refreshed_histories, hidden_conditions,
                 next_state_dist):
        self._refreshed_histories = refreshed_histories
        self._hidden_conditions = hidden_conditions
        self._next_state_dist = next_state_dist

    @property
    def refreshed_histories(self):
        return self._refreshed_histories

    @property
    def hidden_conditions(self):
        return self._hidden_conditions

    @property
    def next_state_dist(self):
        return self._next_state_dist


class GlobalPrior(NonMarkovTransition):
    def __init__(self, prev_local_states,
                 refreshed_local_histories, refreshed_histories,
                 hidden_conditions, pre_flow_dist, next_state_dist):
        assert next_state_dist.reparameterization_type == \
            tfd.FULLY_REPARAMETERIZED
        super(GlobalPrior, self).__init__(
            refreshed_histories, hidden_conditions, next_state_dist
        )
        self._refreshed_local_histories = refreshed_local_histories
        self._prev_local_states = prev_local_states
        self._pre_flow_dist = pre_flow_dist

    @property
    def refreshed_local_histories(self):
        return self._refreshed_local_histories

    @property
    def prev_local_states(self):
        return self._prev_local_states

    @property
    def pre_flow_dist(self):
        return self._pre_flow_dist


class LocalPrior(NonMarkovTransition):
    def __init__(self, global_priors, global_states, global_context,
                 propagated_local_histories, hidden_conditions,
                 pre_flow_dist, next_state_dist):
        assert next_state_dist.reparameterization_type == \
            tfd.FULLY_REPARAMETERIZED
        super(LocalPrior, self).__init__(
            # global_priors.refreshed_local_histories,
            propagated_local_histories,
            hidden_conditions, next_state_dist
        )
        self._global_priors = global_priors
        self._global_states = global_states
        self._global_context = global_context
        self._propagated_histories = propagated_local_histories
        self._pre_flow_dist = pre_flow_dist

    @property
    def global_priors(self):
        return self._global_priors

    @property
    def global_states(self):
        return self._global_states

    @property
    def global_context(self):
        return self._global_context

    @property
    def propagated_histories(self):
        return self._propagated_histories

    @property
    def pre_flow_dist(self):
        return self._pre_flow_dist

    @property
    def full_histories(self):
        return (
            self.global_priors.refreshed_histories,
            self.refreshed_histories
        )


class FactorizedPrior(object):
    def __init__(self, model, graph, external, global_priors):
        self._model = model
        self._graph = graph
        self._external = external
        self._global_priors = global_priors

    @property
    def dim_global_state(self):
        return self._model.dim_latent

    @property
    def dim_local_state(self):
        return self._model.dim_latent

    @property
    def num_nodes(self):
        return self._graph.max_num_nodes

    def sample(self):
        return self._model.sample_latent_states(
            self._graph, self._external, self._global_priors
        )

    def log_prob(self, samples):
        global_states, local_states = samples
        local_priors = self._model.trans_local(
            self._graph, self._external,
            self._global_priors, global_states
        )
        global_prior_dist = self._global_priors.next_state_dist
        local_prior_dist = local_priors.next_state_dist
        return tf.math.add(
            global_prior_dist.log_prob(global_states),
            local_prior_dist.log_prob(local_states)
        )


class NonMarkovModel(object):
    '''
    Generative model for the non-Markovian GNN-SSM:

    p(X_{0:T}, Z_{0:T}) = p(Z_0)p(X_0|Z_0)
                          \\prod _{t=1}^{T}{p(Z_t|Z_{1:t-1})p(X_t|Z_t)}
    '''

    def __init__(
        self,
        dim_observ, dim_latent, dim_hidden,
        dim_mlp, gnn_config,
        dim_global_input=0, dim_local_input=0,
        const_num_nodes=None,
        rnn_num_layers=1,
        init_mix_num_components=1,
        trans_activation="tanh", trans_layer_norm=False,
        trans_mlp_num_layers=1, trans_mix_num_components=1,
        trans_scale_activation="softplus",
        trans_scale_shift=0.0, trans_scale_identical=False,
        trans_ar=False, trans_skip_conn=False,
        trans_global_low_rank=0, trans_local_low_rank=0,
        trans_global_flow=False, trans_flow_num_layers=0,
        trans_flow_mv_factor="qr", trans_flow_skip_conn=True,
        emit_neg_binomial=False, emit_loc_scale_type="normal",
        emit_non_markov=True, emit_identity=False,
        emit_activation="linear", emit_low_rank=0,
        emit_mlp_num_layers=1, emit_mix_num_components=1,
        emit_scale_activation="softplus",
        emit_scale_shift=0.0, emit_scale_identical=True,
        name="non_markov_gen_model"
    ):
        assert not (emit_identity and dim_observ > dim_latent)

        gnn_config = gnn_config.clone()
        gnn_config.readout = None
        gnn_config.dim_input = dim_hidden
        gnn_config.dim_global_state = dim_latent
        dim_readout = 2 * dim_hidden

        with tf.variable_scope(name):
            if trans_skip_conn:
                self._global_skip_conn_layer_norm = layer_norm_1d(
                    dim_latent, trainable=True,
                    name="global_skip_conn_layer_norm"
                )
                self._local_skip_conn_layer_norm = layer_norm_1d(
                    dim_latent, trainable=True,
                    name="local_skip_conn_layer_norm"
                )

            self._trans_gnn = StackedGraphNN(gnn_config, name="trans_gnn")

            if init_mix_num_components > 1:
                self._global_mixture_prior = make_trainable_gmm(
                    dim=dim_latent, num_components=init_mix_num_components,
                    name="trainable_gmm_prior"
                )

            self._global_rnn_cell = util.make_lstm_cells(
                num_layers=rnn_num_layers,
                dim_in=(dim_latent + dim_hidden + dim_global_input),
                cell_size=dim_hidden,
                name="global_rnn_cell"
            )
            self._local_rnn_cell = util.make_lstm_cells(
                num_layers=rnn_num_layers,
                dim_in=(
                    dim_latent + dim_local_input +
                    (dim_observ if trans_ar else 0)
                ),
                cell_size=dim_hidden,
                name="local_rnn_cell"
            )

            self._global_gated_unit = gated_unit(
                dim_i=dim_readout, dim_o=dim_hidden, layer_norm=False,
                name="global_gated_unit"
            )

            self._global_z_plus_h = skip_cond_gated_unit(
                dim_i0=dim_latent, dim_i1=dim_hidden,
                layer_norm=True, name="global_z_plus_h"
            )

            trans_params = dict(
                dim_in=dim_hidden, dim_hid=dim_mlp, dim_out=dim_latent,
                mlp_num_layers=trans_mlp_num_layers,
                loc_activation=trans_activation,
                loc_layer_norm=trans_layer_norm,
                scale_activation=trans_scale_activation,
                scale_shift=trans_scale_shift,
                scale_identical=trans_scale_identical
            )

            if trans_global_low_rank > 0:
                self._global_trans_dist = mlp_low_rank_normal(
                    **trans_params, cov_rank=trans_global_low_rank,
                    name="global_trans_dist"
                )
            else:
                self._global_trans_dist = mlp_mix_diag_normal(
                    **trans_params,
                    mix_num_components=trans_mix_num_components,
                    name="global_trans_dist"
                )

            if trans_local_low_rank > 0:
                self._local_trans_dist = mlp_low_rank_normal(
                    **trans_params, cov_rank=trans_local_low_rank,
                    name="local_trans_dist"
                )
            else:
                self._local_trans_dist = mlp_mix_diag_normal(
                    **trans_params,
                    mix_num_components=trans_mix_num_components,
                    name="local_trans_dist"
                )

            if trans_flow_num_layers > 0:
                if trans_global_flow:
                    self._global_trans_flow_components = init_real_nvp(
                        num_layers=trans_flow_num_layers,
                        dim_latent=dim_latent,
                        dim_context=dim_hidden, dim_mlp=dim_mlp,
                        conv_1x1_factor=trans_flow_mv_factor,
                        name="global_trans_flow"
                    )
                self._local_trans_flow_components = init_perm_equiv_flow(
                    num_layers=trans_flow_num_layers,
                    dim_latent=dim_latent, dim_context=dim_hidden,
                    nvp_gnn_config=gnn_config.single_layer,
                    conv_1x1_factor=trans_flow_mv_factor,
                    name="local_trans_perm_equiv_flow"
                )

            assert not (emit_identity and emit_neg_binomial)
            assert not (emit_identity and emit_mix_num_components > 1)
            assert not (emit_identity and emit_non_markov)
            assert not (emit_identity and emit_activation != "linear")

            if emit_non_markov:
                self._emit_z_plus_h = skip_cond_gated_unit(
                    dim_i0=(2 * dim_latent), dim_i1=(2 * dim_hidden),
                    layer_norm=False, name="z_plus_h"
                )

            emit_dist_params = dict(
                dim_in=(2 * dim_latent), dim_hid=dim_mlp, dim_out=dim_observ,
                mlp_num_layers=emit_mlp_num_layers,
                name="emit_dist"
            )
            emit_loc_scale_dist_params = dict(
                **emit_dist_params,
                loc_activation=emit_activation,
                loc_layer_norm=False,
                scale_activation=emit_scale_activation,
                scale_shift=emit_scale_shift,
                scale_identical=emit_scale_identical
            )
            if emit_identity:
                self._local_emit_dist = identity_diag_normal(
                    dim_in=dim_latent, dim_out=dim_observ,
                    scale_activation=emit_scale_activation,
                    scale_shift=emit_scale_shift,
                    name="emit_dist"
                )
            elif emit_neg_binomial:
                assert dim_observ == 1
                assert emit_mix_num_components == 1  # TODO
                self._local_emit_dist = mlp_mix_neg_binomial(
                    mix_num_components=emit_mix_num_components,
                    **emit_dist_params
                )
            elif emit_low_rank > 0:
                self._local_emit_dist = mlp_low_rank_normal(
                    **emit_loc_scale_dist_params,
                    cov_rank=emit_low_rank
                )
            else:
                builder = get_mlp_mix_loc_scale_builder(emit_loc_scale_type)
                self._local_emit_dist = builder(
                    **emit_loc_scale_dist_params,
                    mix_num_components=emit_mix_num_components
                )

        self._dim_latent = dim_latent
        self._dim_observ = dim_observ
        self._dim_hidden = dim_hidden
        self._dim_global_input = dim_global_input
        self._dim_local_input = dim_local_input
        self._const_num_nodes = const_num_nodes
        self._gnn_config = gnn_config
        self._rnn_num_layers = rnn_num_layers
        self._num_trans_modes = trans_mix_num_components
        self._num_emit_modes = emit_mix_num_components
        self._use_mixture_prior = (init_mix_num_components > 1)
        self._trans_ar = trans_ar
        self._trans_skip_conn = trans_skip_conn
        self._trans_global_flow = trans_global_flow
        self._trans_flow_num_layers = trans_flow_num_layers
        self._trans_flow_skip_conn = trans_flow_skip_conn
        self._emit_identity = emit_identity
        self._emit_non_markov = emit_non_markov

        if emit_identity:
            self._state_mask = tf.concat([
                tf.zeros([dim_observ]),
                tf.ones([dim_latent - dim_observ])
            ], axis=-1)
        else:
            self._state_mask = tf.ones([dim_latent])

    @property
    def dim_observ(self):
        return self._dim_observ

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def dim_hidden(self):
        return self._dim_hidden

    @property
    def dim_full_hidden(self):
        return 2 * self.rnn_num_layers * self.dim_hidden

    @property
    def dim_global_input(self):
        return self._dim_global_input

    @property
    def dim_local_input(self):
        return self._dim_local_input

    @property
    def const_num_nodes(self):
        assert self._const_num_nodes is not None
        return self._const_num_nodes

    @property
    def gnn_config(self):
        return self._gnn_config

    @property
    def rnn_num_layers(self):
        return self._rnn_num_layers

    @property
    def use_mixture_prior(self):
        return self._use_mixture_prior

    @property
    def trans_ar(self):
        return self._trans_ar

    @property
    def trans_skip_conn(self):
        return self._trans_skip_conn

    @property
    def trans_global_flow(self):
        return self._trans_global_flow

    @property
    def trans_flow_num_layers(self):
        return self._trans_flow_num_layers

    @property
    def trans_flow_skip_conn(self):
        return self._trans_flow_skip_conn

    @property
    def global_trans_flow_components(self):
        return self._global_trans_flow_components

    @property
    def local_trans_flow_components(self):
        return self._local_trans_flow_components

    @property
    def emit_non_markov(self):
        return self._emit_non_markov

    @property
    def emit_identity(self):
        return self._emit_identity

    @property
    def state_mask(self):
        return self._state_mask

    @property
    def num_trans_modes(self):
        return self._num_trans_modes

    def _concat_with_inputs(self, context, inputs):
        if inputs is None:
            return context
        if inputs.shape.ndims < context.shape.ndims:
            inputs = tf.math.add(
                inputs, tf.zeros(tf.stack([
                    *tf.unstack(tf.shape(context)[:-1]), util.dim(inputs)
                ]))
            )
        return tf.concat([context, inputs], axis=-1)

    def _broadcast_and_concat(self, global_context, local_context):
        global_context = tf.expand_dims(global_context, axis=-2)
        global_context = tf.math.add(
            global_context, tf.zeros(tf.stack([
                *tf.unstack(tf.shape(local_context)[:-1]),
                util.dim(global_context)
            ]))
        )
        return tf.concat([global_context, local_context], axis=-1)

    def init_global_histories(self, batch_shape):
        '''
        Returns:
          global_histories: A (..., dH) Tensor.
        '''
        return _init_zeros(batch_shape, self.dim_full_hidden)

    def init_local_histories(self, prefix_shape):
        '''
        Returns:
          local_histories: A (..., N, dH) Tensor.
        '''
        return _init_zeros(prefix_shape, self.dim_full_hidden)

    def extract_rnn_output(self, histories):
        return util.extract_rnn_output(
            histories, self.rnn_num_layers, util.LSTM
        )

    def extract_all_rnn_output(self, histories):
        return (
            self.extract_rnn_output(histories[0]),
            self.extract_rnn_output(histories[1])
        )

    def readout(self, graph, local_histories):
        '''
        Args:
          local_histories: A (..., N, dH) Tensor.

        Returns:
          summaries: A (..., 2*dh) Tensor.
        '''
        return self._global_gated_unit(
            readout_mean_max(self.gnn_config, graph, local_histories)
        )

    def refresh_global_histories(self, graph, external,
                                 global_histories, global_states, readouts):
        '''
        Args:
          histories: A (..., dH) Tensor which summrizes Z_{1:t-1}.
          new_states: A (..., dz) Tensor.
          readouts: A (..., dh) Tensor.

        Returns:
          new_histories: A (..., dH) Tensor.
        '''
        return refresh_histories_using_lstm(
            self._global_rnn_cell, global_histories,
            self._concat_with_inputs(
                tf.concat([global_states, readouts], axis=-1),
                external.global_inputs
            )
        )

    def refresh_local_histories(self, graph, external,
                                local_histories, local_states, observations):
        '''
        Args:
          histories: A (..., dH) Tensor that summrizes Z_{1:t-1}.
          new_states: A (..., dz) Tensor.

        Returns:
          new_histories: A (..., dH) Tensor.
        '''
        if self.trans_ar:
            local_states = self._concat_with_inputs(local_states, observations)
        return refresh_histories_using_lstm(
            self._local_rnn_cell, local_histories,
            self._concat_with_inputs(local_states, external.local_inputs)
        )

    def propagate_local_histories(
            self, graph, global_context, local_histories):
        '''
        Args:
          global_context: A (..., dz) Tensor.
          local_histories: A (..., N, dH) Tensor.

        Returns:
          correlated_context: A (..., N, dH) Tensor.
        '''
        correlated_histories = self._trans_gnn(
            graph=graph, states=local_histories,
            global_states=global_context
        )
        mask = tf.expand_dims(graph.center_mask, axis=-1)
        return util.select(mask, correlated_histories, local_histories)

    def global_prior(self, graph, external, batch_shape):
        '''
        Returns:
          A distribution with batch shape (shape) and event shape (dz).
        '''
        del graph, external

        if self.use_mixture_prior:
            return util.broadcast_gmm(self._global_mixture_prior, batch_shape)

        full_shape = tf.stack([*tf.unstack(batch_shape), self.dim_latent])
        return tfd.MultivariateNormalDiag(
            loc=tf.zeros(full_shape), scale_diag=tf.ones(full_shape)
        )

    def init_global(self, graph, external, num_samples, prefix_shape):
        '''
        Args:
          prefix_shape: A tuple, (..., B, N).

        Returns:
          global_prior: A GlobalPrior object.
        '''
        prefix = tf.stack([num_samples, *tf.unstack(prefix_shape)])

        global_context = tf.zeros(tf.stack([
            *tf.unstack(prefix[:-1]), self.dim_hidden
        ]))

        global_prior_dist = self.global_prior(graph, external, prefix[:-1])

        pre_flow_dist = global_prior_dist
        if self.trans_global_flow and self.trans_flow_num_layers > 0:
            global_prior_dist = real_nvp_wrapper(
                components=self.global_trans_flow_components,
                context=global_context,
                base_dist=global_prior_dist,
                skip_conn=self.trans_flow_skip_conn
            )

        return GlobalPrior(
            prev_local_states=tf.zeros(tf.stack([
                *tf.unstack(prefix), self.dim_latent
            ])),
            refreshed_local_histories=self.init_local_histories(prefix),
            refreshed_histories=self.init_global_histories(prefix[:-1]),
            hidden_conditions=global_context,
            pre_flow_dist=pre_flow_dist,
            next_state_dist=global_prior_dist
        )

    def trans_global(self, graph, external, histories, states, observations):
        '''
        Z_{1:t} -> Z_{t+1}  <=> (H_{t-1}, Z_t) -> Z_{t+1}

        Args:
          graph: A RuntimeGraph object.
          inputs: Optional. A (..., [N, ]di) Tensor.
          histories: A 2-ary tuple:
            - global_histories: A (..., dH) Tensor.
            - local_histories: A (..., N, dH) Tensor.
          states: A 2-ary tuple:
            - global_states: A (..., dz) Tensor.
            - local_states: A (..., N, dz) Tensor.
          observations: A (..., N, dx) Tensor.

        Returns:
          global_transition: A GlobalTransition object.
        '''
        global_histories, local_histories = histories
        global_states, local_states = states

        new_local_histories = self.refresh_local_histories(
            graph, external, local_histories, local_states, observations
        )
        new_global_histories = self.refresh_global_histories(
            graph, external, global_histories, global_states,
            self.readout(graph, new_local_histories)
        )
        new_global_context = self.extract_rnn_output(new_global_histories)
        new_global_state_dist = self._global_trans_dist(new_global_context)

        if self.trans_skip_conn:
            assert type(new_global_state_dist) is util.PartialLocScaleDist
            new_global_state_dist.loc = self._global_skip_conn_layer_norm(
                tf.math.add(new_global_state_dist.loc, global_states)
            )

        if type(new_global_state_dist) is util.PartialLocScaleDist:
            new_global_state_dist = new_global_state_dist.build()

        pre_flow_dist = new_global_state_dist

        if self.trans_global_flow and self.trans_flow_num_layers > 0:
            new_global_state_dist = real_nvp_wrapper(
                components=self.global_trans_flow_components,
                context=new_global_context,
                base_dist=new_global_state_dist,
                skip_conn=self.trans_flow_skip_conn
            )

        return GlobalPrior(
            prev_local_states=local_states,
            refreshed_local_histories=new_local_histories,
            refreshed_histories=new_global_histories,
            hidden_conditions=new_global_context,
            pre_flow_dist=pre_flow_dist,
            next_state_dist=new_global_state_dist,
        )

    def trans_local(self, graph, external, global_priors, global_states):
        '''
        Z_{1:t} -> Z_{t+1}  <=> (H_{t-1}, Z_t) -> Z_{t+1}

        Args:
          graph: A RuntimeGraph object.
          external: An External Tensor.
          global_transition: A GlobalTransition object.
          global_states: A (..., dz) Tensor.

        Returns:
          local_transition: A LocalTransition object.
        '''
        new_global_context = self._global_z_plus_h(
            global_states, global_priors.hidden_conditions
        )
        correlated_histories = self.propagate_local_histories(
            graph, new_global_context,
            global_priors.refreshed_local_histories
        )
        new_local_context = self.extract_rnn_output(correlated_histories)
        new_local_state_dist = self._local_trans_dist(new_local_context)

        if self.trans_skip_conn:
            assert type(new_local_state_dist) is util.PartialLocScaleDist
            new_local_state_dist.loc = self._local_skip_conn_layer_norm(
                tf.math.add(
                    global_priors.prev_local_states,
                    new_local_state_dist.loc
                )
            )

        if type(new_local_state_dist) is util.PartialLocScaleDist:
            new_local_state_dist = new_local_state_dist.build()

        pre_flow_dist = new_local_state_dist

        new_local_state_dist = tfd.Independent(
            distribution=new_local_state_dist,
            reinterpreted_batch_ndims=1,
            name="indep_" + new_local_state_dist.name
        )

        # TODO: mask
        if self.trans_flow_num_layers > 0:
            new_local_state_dist = perm_equiv_flow_wrapper(
                components=self.local_trans_flow_components,
                graph=graph, const_num_nodes=self.const_num_nodes,
                global_context=new_global_context,
                local_context=new_local_context,
                base_dist=new_local_state_dist,
                skip_conn=self.trans_flow_skip_conn
            )

        return LocalPrior(
            global_priors=global_priors,
            global_states=global_states,
            global_context=new_global_context,
            propagated_local_histories=correlated_histories,
            hidden_conditions=new_local_context,
            pre_flow_dist=pre_flow_dist,
            next_state_dist=new_local_state_dist
        )

    def emit(self, histories, states):
        '''
        Z_{1:t} -> X_t

        Args:
          histories: A 2-ary tuple:
            - global_histories: A (..., dH) Tensor.
            - local_histories: A (..., N, dH) Tensor.
          states: A 2-ary tuple:
            - global_states: A (..., dz) Tensor.
            - local_states: A (..., N, dz) Tensor.

        Returns:
          A distribution with batch shape (..., N) and event shape (dx).
        '''
        global_histories, local_histories = histories

        global_context = self.extract_rnn_output(global_histories)
        local_context = self.extract_rnn_output(local_histories)
        context = (global_context, local_context)
        concat_context = self._broadcast_and_concat(*context)
        concat_states = self._broadcast_and_concat(*states)

        emit_context = self._emit_z_plus_h(concat_states, concat_context) \
            if self.emit_non_markov else concat_states

        dist = self._local_emit_dist(emit_context)
        if type(dist) is util.PartialLocScaleDist:
            dist = dist.build()
        return dist

    def factorized_prior(self, graph, external, global_priors):
        return FactorizedPrior(self, graph, external, global_priors)

    def sample_latent_states(self, graph, external, global_priors):
        new_global_states = global_priors.next_state_dist.sample(1)[0]
        local_priors = self.trans_local(
            graph, external, global_priors, new_global_states
        )
        new_local_states = local_priors.next_state_dist.sample(1)[0]
        log_probs = tf.math.add(
            global_priors.next_state_dist.log_prob(new_global_states),
            local_priors.next_state_dist.log_prob(new_local_states)
        )
        samples = (new_global_states, new_local_states)
        histories = local_priors.full_histories
        return samples, histories, log_probs

    def recover_histories_if_stale(self, graph, external, global_priors,
                                   histories, states):
        if histories is not None:
            return histories
        global_states, _ = states
        local_priors = self.trans_local(
            graph, external, global_priors, global_states
        )
        return local_priors.full_histories


def predict(model, graph, external,
            end_histories, end_states, end_observations, horizon):
    '''
    Args:
      model: The generative model.
      graph: A RuntimeGraph object.
      end_histories: A tuple of Tensors: (S, B, dh) and (S, B, N, dh).
      end_states: A tuple of Tensors: (S, B, dz) and (S, B, N, dz).
      end_observations: A (B, N, dx) Tensor.
      horizon: A scalar.

    Returns:
      predictions: A (S, H, B, N, dx) Tensor.
      log_probs: A (S, H, B) Tensor.
    '''
    _, end_local_states = end_states
    num_samples = tf.shape(end_local_states)[0]

    def cond(t, *unused_args):
        return tf.less(t, horizon)

    def body(t, histories, states, observations, log_probs_acc,
             locs_array, scales_array, log_probs_array):
        global_priors = model.trans_global(
            graph, external.current(t),
            histories, states, observations
        )
        new_global_states = global_priors.next_state_dist.sample(1)[0]
        local_priors = model.trans_local(
            graph, external.current(t),
            global_priors, new_global_states
        )
        new_local_states = local_priors.next_state_dist.sample(1)[0]
        new_log_probs_acc = tf.math.add(
            log_probs_acc, tf.math.add(
                global_priors.next_state_dist.log_prob(new_global_states),
                local_priors.next_state_dist.log_prob(new_local_states)
            )
        )

        new_histories = local_priors.full_histories
        new_states = (new_global_states, new_local_states)
        likelihood = model.emit(new_histories, new_states)

        new_locs_array = locs_array.write(t, likelihood.mean())
        new_scales_array = scales_array.write(t, likelihood.stddev())
        new_log_probs_array = log_probs_array.write(t, new_log_probs_acc)

        return (
            t + 1, new_histories, new_states, likelihood.mean(),
            new_log_probs_acc,
            new_locs_array, new_scales_array, new_log_probs_array
        )

    broadcast_end_observations = tf.tile(
        tf.expand_dims(end_observations, axis=0),
        tf.stack([num_samples] + ([1] * end_observations.shape.ndims))
    )

    t0 = tf.constant(0)
    init_log_probs_acc = tf.zeros(tf.shape(end_local_states)[:-2])
    init_locs_array = tf.TensorArray(tf.float32, size=horizon)
    init_scales_array = tf.TensorArray(tf.float32, size=horizon)
    init_log_probs_array = tf.TensorArray(tf.float32, size=horizon)

    _, _, _, _, _, locs_array, scales_array, log_probs_array = tf.while_loop(
        cond, body,
        [
            t0, end_histories, end_states, broadcast_end_observations,
            init_log_probs_acc,
            init_locs_array, init_scales_array, init_log_probs_array
        ]
    )
    locs, scales = locs_array.stack(), scales_array.stack()
    log_probs = log_probs_array.stack()

    assert locs.shape.ndims == 5
    assert scales.shape.ndims == 5
    assert log_probs.shape.ndims == 3

    static_batch_shape = end_local_states.shape.as_list()[:-1]
    locs.set_shape([horizon, *static_batch_shape, None])
    scales.set_shape([horizon, *static_batch_shape, None])
    log_probs.set_shape([horizon, *static_batch_shape[:-1]])
    # (H, S, B, N, dx) -> (S, H, B, N, dx)
    locs = tf.transpose(locs, perm=[1, 0, 2, 3, 4])
    scales = tf.transpose(scales, perm=[1, 0, 2, 3, 4])
    # (H, S, B) -> (S, H, B)
    log_probs = tf.transpose(log_probs, perm=[1, 0, 2])

    return locs, scales, log_probs


# TODO

def preview(model, graph, external, histories, states, num_preview_steps):
    for t in range(num_preview_steps - 1):
        priors = model.trans(
            graph=graph, histories=histories, states=states,
            inputs=external.current(t)
        )
        next_states_dist = priors.next_state_dist
        histories = priors.refreshed_histories
        states = next_states_dist.sample(1)[0]
    return tf.concat([histories, states], axis=-1)


def init_preview_array(
        num_time_steps, num_preview_steps,
        model, graph, external, initial_states, initial_histories):
    preview_array = tf.TensorArray(
        tf.float32, size=(num_time_steps + num_preview_steps)
    )
    zero_previews = tf.concat(
        [tf.zeros_like(initial_histories), tf.zeros_like(initial_states)],
        axis=-1
    )
    for k in range(num_preview_steps):
        preview_array = preview_array.write(k, zero_previews)
    preview_array = preview_array.write(
        num_preview_steps,
        preview(
            model, graph, external, initial_states, initial_histories,
            num_preview_steps
        )
    )
    return preview_array
