from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import util
from util import mlp_diag_normal, mlp_low_rank_normal
from util import make_lstm_cells
from util import skip_cond_gated_unit, gated_unit
from util import extract_rnn_output, layer_norm_1d
from flow import init_real_nvp, real_nvp_wrapper
from flow import init_perm_equiv_flow, perm_equiv_flow_wrapper
from gnn import RecurrentGraphNN

import functools

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

SUMMARIZER_LSTM = "lstm"
SUMMARIZER_RGNN = "rgnn"


def collect_persistent_states(
        belief_states, latent_histories, latent_states):
    states_tuple = (belief_states, latent_histories, latent_states)
    return states_tuple


def _init_summ_like_seq(sequence, dim):
    return tf.zeros(tf.stack([
        *tf.unstack(tf.shape(sequence)[1:-1]), dim
    ]))


def _init_summ_like(states, dim):
    return tf.zeros(tf.stack([
        *tf.unstack(tf.shape(states)[:-1]), dim
    ]))


def summarize_forward_using_lstm(
        lstm_cell, graph, sequence, initial_states):
    '''
    Args:
      sequence: A (T, B, N, dx) Tensor.
      initial_states: A (B, N, dh) Tensor.

    Returns:
      summaries: A (T, B, N, dh) Tensor.
      last_states: A (B, N, dh) Tensor.
    '''
    del graph

    num_steps = tf.shape(sequence)[0]

    def cond(t, *unused_args):
        return tf.less(t, num_steps)

    def body(t, states, summaries):
        _, new_states = lstm_cell(sequence[t], states)
        new_summaries = summaries.write(t, util.concat_rnn_states(new_states))
        t = tf.add(t, 1)
        return t, new_states, new_summaries

    t0 = tf.constant(0)
    initial_lstm_states = util.pack_rnn_states(lstm_cell, initial_states)
    initial_summaries = tf.TensorArray(tf.float32, size=num_steps)

    _, last_lstm_states, summaries = tf.while_loop(
        cond, body,
        [t0, initial_lstm_states, initial_summaries],
    )

    last_states = util.concat_rnn_states(last_lstm_states)
    summaries = summaries.stack()
    summaries.set_shape([
        *sequence.shape.as_list()[:-1], None
    ])
    return summaries, last_states


def summarize_forward_using_recurrent_gnn(
        rgnn, graph, sequence, initial_states):
    last_states, summaries = rgnn(graph, initial_states, sequence)
    return summaries, last_states


def summarize_backward_using_lstm(
        lstm_cell, graph, sequence, initial_states):
    reversed_sequence = tf.reverse(sequence, [0])
    summaries, last_states = summarize_forward_using_lstm(
        lstm_cell, graph.reverse(), reversed_sequence, initial_states
    )
    return tf.reverse(summaries, [0]), last_states


def summarize_backward_using_recurrent_gnn(
        rgnn, graph, sequence, initial_states):
    reversed_sequence = tf.reverse(sequence, [0])
    summaries, last_states = summarize_forward_using_recurrent_gnn(
        rgnn, graph.reverse(), reversed_sequence, initial_states
    )
    return tf.reverse(summaries, [0]), last_states


def refresh_lstm_states(lstm_cell, graph, flat_latest_states, new_inputs):
    del graph
    lstm_states = util.pack_rnn_states(lstm_cell, flat_latest_states)
    _, new_lstm_states = lstm_cell(new_inputs, lstm_states)
    return util.concat_rnn_states(new_lstm_states)


def refresh_rgnn_states(rgnn_cell, graph, flat_latest_states, new_inputs):
    new_input_seqs = tf.expand_dims(new_inputs, axis=0)
    new_flat_states, _ = rgnn_cell(graph, flat_latest_states, new_input_seqs)
    return new_flat_states


class FactorizedProposalDistribution(object):
    def __init__(self, model, graph, external, observations, conditions,
                 global_priors, global_proposal_dist, local_proposal_fn,
                 context):
        self._model = model
        self._graph = graph
        self._external = external
        self._observations = observations
        self._conditions = conditions
        self._global_priors = global_priors
        self._global_proposal_dist = global_proposal_dist
        self._local_proposal_fn = local_proposal_fn
        self._context = context

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
        global_prior_dist = self._global_priors.next_state_dist
        global_proposal_dist = self._global_proposal_dist
        global_states = global_proposal_dist.sample(1)[0]

        local_priors = self._model.trans_local(
            self._graph, self._external,
            self._global_priors, global_states
        )
        local_prior_dist = local_priors.next_state_dist
        local_proposal_dist = self._local_proposal_fn(
            self._graph, self._external, local_priors,
            self._observations, self._conditions, self._context
        )
        local_states = local_proposal_dist.sample(1)[0]

        log_proposal_probs = tf.math.add(
            global_proposal_dist.log_prob(global_states),
            local_proposal_dist.log_prob(local_states)
        )
        log_prior_probs = tf.math.add(
            global_prior_dist.log_prob(global_states),
            local_prior_dist.log_prob(local_states)
        )
        assert log_proposal_probs.shape.ndims == log_prior_probs.shape.ndims
        assert log_prior_probs.shape.ndims == global_states.shape.ndims - 1

        samples = (global_states, local_states)
        histories = local_priors.full_histories
        return (
            samples, histories, log_proposal_probs, log_prior_probs
        )

    def log_prob(self, samples):
        global_states, local_states = samples
        local_priors = self._model.trans_local(
            self._graph, self._external,
            self._global_priors, global_states
        )
        local_proposal_dist = self._local_proposal_fn(
            self._graph, self._external, local_priors,
            self._observations, self._conditions, self._context
        )
        return tf.math.add(
            self._global_proposal_dist.log_prob(global_states),
            local_proposal_dist.log_prob(local_states)
        )


class FactorizedWrapper(object):
    def __init__(self, proposal):
        self._proposal = proposal

    @property
    def dim_hidden(self):
        return self._proposal.dim_hidden

    @property
    def dim_latent(self):
        return self._proposal.dim_latent

    @property
    def rnn_num_layers(self):
        return self._proposal.rnn_num_layers

    def summarize_forward(self, *args, **kwargs):
        return self._proposal.summarize_forward(*args, **kwargs)

    def summarize_backward(self, *args, **kwargs):
        return self._proposal.summarize_backward(*args, **kwargs)

    def extract_summaries(self, *args, **kwargs):
        return self._proposal.extract_summaries(*args, **kwargs)

    def refresh_beliefs(self, *args, **kwargs):
        return self._proposal.refresh_beliefs(*args, **kwargs)

    def concat_conditions(self, *args, **kwargs):
        return self._proposal.concat_conditions(*args, **kwargs)

    def perturb(self, *args, **kwargs):
        return self._proposal.perturb(*args, **kwargs)

    def propose(self, graph, external, global_priors,
                observations, conditions, context=None):
        global_proposal_dist = self._proposal.propose_global(
            graph, external, global_priors,
            observations, conditions, context
        )
        return FactorizedProposalDistribution(
            model=self._proposal.model,
            graph=graph, external=external,
            observations=observations, conditions=conditions,
            global_priors=global_priors,
            global_proposal_dist=global_proposal_dist,
            local_proposal_fn=self._proposal.propose_local,
            context=context
        )


class ProposalContext(object):
    def __init__(self, t, length):
        self._t = t
        self._length = length
        self._use_proposal = None

    @property
    def t(self):
        return self._t

    @property
    def length(self):
        return self._length

    @property
    def use_proposal(self):
        return self._use_proposal

    @use_proposal.setter
    def use_proposal(self, value):
        self._use_proposal = value


class ProposalBuilder(object):
    def __init__(self, model, dim_mlp, gnn_config,
                 summarize_unit=SUMMARIZER_LSTM,
                 loc_activation="linear", loc_layer_norm=False,
                 scale_activation="softplus",
                 scale_shift=0.0001, scale_identical=False,
                 scale_identity=0.0005,
                 rnn_num_layers=1, mlp_num_layers=1,
                 global_low_rank=0, local_low_rank=0,
                 global_flow=False, flow_num_layers=0,
                 flow_mv_factor="qr", flow_skip_conn=True,
                 reuse_gen_flow=False,
                 use_belief=True, use_lookahead=False,
                 use_skip_conn=True, use_gated_adder=False,
                 denoising=True, noise_scale=0.1,
                 name="ProposalBuilder"):
        dim_observ = model.dim_observ
        dim_latent = model.dim_latent
        dim_hidden = model.dim_hidden
        dim_global_input = model.dim_global_input
        dim_local_input = model.dim_local_input
        gnn_config = gnn_config.clone()
        gnn_config.dim_input = dim_hidden
        gnn_config.dim_global_state = gnn_config.dim_readout
        dim_observ_and_input = dim_observ + dim_local_input + dim_global_input

        with tf.variable_scope(name):
            if use_skip_conn:
                self._global_skip_conn_layer_norm = layer_norm_1d(
                    dim_latent, trainable=True,
                    name="global_skip_conn_layer_norm"
                )
                self._local_skip_conn_layer_norm = layer_norm_1d(
                    dim_latent, trainable=True,
                    name="local_skip_conn_layer_norm"
                )

            if not (summarize_unit == SUMMARIZER_LSTM or
                    summarize_unit == SUMMARIZER_RGNN):
                raise ValueError("Unknown name: {}".format(summarize_unit))

            # Summarize X_{1:t} into B_t & X_{t:T} into L_t
            if summarize_unit == SUMMARIZER_LSTM:
                self._forward_lstm_cell = make_lstm_cells(
                    num_layers=rnn_num_layers,
                    dim_in=dim_observ_and_input,
                    cell_size=dim_hidden,
                    name="forward_lstm_cell"
                )
                self._summarize_forward_fn = functools.partial(
                    summarize_forward_using_lstm,
                    self._forward_lstm_cell
                )
                self._refresh_belief_fn = functools.partial(
                    refresh_lstm_states, self._forward_lstm_cell
                )
            if summarize_unit == SUMMARIZER_LSTM or (
                    summarize_unit == SUMMARIZER_RGNN and not use_lookahead):
                self._backward_lstm_cell = make_lstm_cells(
                    num_layers=rnn_num_layers,
                    dim_in=dim_observ_and_input,
                    cell_size=dim_hidden,
                    name="backward_lstm_cell"
                )
                self._summarize_backward_fn = functools.partial(
                    summarize_backward_using_lstm,
                    self._backward_lstm_cell
                )
            if summarize_unit == SUMMARIZER_RGNN:
                self._forward_recurrent_gnn = RecurrentGraphNN(
                    gnn_config, dim_input=dim_observ_and_input,
                    name="forward_recurrent_gnn"
                )
                self._summarize_forward_fn = functools.partial(
                    summarize_forward_using_recurrent_gnn,
                    self._forward_recurrent_gnn
                )
                self._refresh_belief_fn = functools.partial(
                    refresh_rgnn_states, self._forward_recurrent_gnn
                )
            if summarize_unit == SUMMARIZER_RGNN and use_lookahead:
                self._backward_recurrent_gnn = RecurrentGraphNN(
                    gnn_config, dim_input=dim_observ_and_input,
                    name="backward_recurrent_gnn"
                )
                self._summarize_backward_fn = functools.partial(
                    summarize_backward_using_recurrent_gnn,
                    self._backward_recurrent_gnn
                )

            if flow_num_layers > 0:
                if reuse_gen_flow:
                    if global_flow:
                        self._global_flow_components = \
                            model.global_trans_flow_components
                    self._local_flow_components = \
                        model.local_trans_flow_components
                else:
                    if global_flow:
                        self._global_flow_components = init_real_nvp(
                            num_layers=flow_num_layers,
                            dim_latent=dim_latent,
                            dim_context=dim_hidden, dim_mlp=dim_mlp,
                            conv_1x1_factor=flow_mv_factor,
                            name="proposal_global_flow"
                        )
                    self._local_flow_components = init_perm_equiv_flow(
                        num_layers=flow_num_layers,
                        dim_latent=dim_latent, dim_context=dim_hidden,
                        nvp_gnn_config=gnn_config.single_layer,
                        conv_1x1_factor=flow_mv_factor,
                        name="proposal_local_perm_equiv_flow"
                    )

            if use_belief or use_lookahead:
                use_both = use_belief and use_lookahead
                dim_i1 = (dim_hidden * (2 if use_both else 1))
            else:
                dim_i1 = dim_observ_and_input

            self._global_skip_cond_gated_unit = skip_cond_gated_unit(
                dim_i0=dim_hidden,
                dim_i1=(2 * dim_i1 + dim_global_input),
                layer_norm=True,
                name="global_skip_cond_gated_unit"
            )
            self._local_skip_cond_gated_unit = skip_cond_gated_unit(
                dim_i0=dim_hidden,
                dim_i1=(dim_i1 + dim_local_input + dim_observ),
                layer_norm=True,
                name="local_skip_cond_gated_unit"
            )

            if use_gated_adder:
                self._global_gated_unit = gated_unit(
                    dim_i=dim_hidden, dim_o=dim_latent,
                    name="global_gated_unit"
                )
                self._local_gated_unit = gated_unit(
                    dim_i=dim_hidden, dim_o=dim_latent,
                    name="local_gated_unit"
                )

            proposal_params = dict(
                dim_hid=dim_mlp,
                dim_out=dim_latent,
                mlp_num_layers=mlp_num_layers,
                loc_activation=loc_activation,
                loc_layer_norm=loc_layer_norm,
                scale_activation=scale_activation,
                scale_shift=scale_shift,
                scale_identical=scale_identical
            )
            global_proposal_params = dict(
                dim_in=dim_hidden, **proposal_params
            )
            local_proposal_params = dict(
                dim_in=dim_hidden + dim_observ, **proposal_params
            )

            if global_low_rank > 0:
                self._global_mlp_normal = mlp_low_rank_normal(
                    **global_proposal_params, cov_rank=global_low_rank,
                    name="global_mlp_low_rank_normal"
                )
            else:
                self._global_mlp_normal = mlp_diag_normal(
                    **global_proposal_params,
                    name="global_mlp_diag_normal"
                )

            if local_low_rank > 0:
                self._local_mlp_normal = mlp_low_rank_normal(
                    **local_proposal_params, cov_rank=local_low_rank,
                    name="local_mlp_low_rank_normal"
                )
            else:
                self._local_mlp_normal = mlp_diag_normal(
                    **local_proposal_params,
                    name="local_mlp_diag_normal"
                )

        self._model = model
        self._dim_hidden = dim_hidden
        self._dim_latent = dim_latent
        self._rnn_num_layers = rnn_num_layers
        self._global_flow = global_flow
        self._flow_num_layers = flow_num_layers
        self._flow_skip_conn = flow_skip_conn
        self._summarize_unit = summarize_unit
        self._use_lookahead = use_lookahead
        self._use_belief = use_belief
        self._use_skip_conn = use_skip_conn
        self._use_gated_adder = use_gated_adder
        self._scale_identity = scale_identity
        self._denoising = denoising
        self._noise_scale = noise_scale

    @property
    def model(self):
        return self._model

    @property
    def dim_hidden(self):
        return self._dim_hidden

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def dim_belief_state(self):
        return self.rnn_num_layers * 2 * self.dim_hidden

    @property
    def rnn_num_layers(self):
        return self._rnn_num_layers

    @property
    def global_flow(self):
        return self._global_flow

    @property
    def flow_num_layers(self):
        return self._flow_num_layers

    @property
    def flow_skip_conn(self):
        return self._flow_skip_conn

    @property
    def use_lookahead(self):
        return self._use_lookahead

    @property
    def use_belief(self):
        return self._use_belief

    @property
    def use_bi_rnn(self):
        return self._use_belief and self._use_lookahead

    @property
    def use_skip_conn(self):
        return self._use_skip_conn

    @property
    def use_gated_adder(self):
        return self._use_gated_adder

    @property
    def denoising(self):
        return self._denoising

    @property
    def noise_scale(self):
        return self._noise_scale

    def _concat_input_and_observ(self, external, observations):
        global_inputs = external.global_inputs
        local_inputs = external.local_inputs
        result = observations

        if local_inputs is not None:
            result = tf.concat([result, local_inputs], axis=-1)

        if global_inputs is None:
            return result
        # ([T, ]B, dt) -> ([T, ]B, N, dt)
        global_inputs = tf.expand_dims(global_inputs, axis=-2)
        broadcast_global_inputs = tf.math.add(
            global_inputs, tf.zeros(tf.stack([
                *tf.unstack(tf.shape(observations)[:-1]),
                self.model.dim_global_input
            ]))
        )
        return tf.concat([result, broadcast_global_inputs], axis=-1)

    def _summarize_with_fn(self, fn):
        def summarize(graph, external, sequence, initial_states=None):
            '''
            Summarize the observations into belief states.

            Args:
              graph: A RuntimeGraph object.
              sequence: A (T, B, N, dx) Tensor.

            Returns:
              summaries: A (T, B, N, dh) Tensor.
            '''
            sequence = self._concat_input_and_observ(external, sequence)
            initial_states = initial_states if initial_states is not None \
                else _init_summ_like_seq(sequence, self.dim_belief_state)
            per_step_states, final_states = fn(
                graph=graph, sequence=sequence,
                initial_states=initial_states
            )
            summaries = self.extract_summaries(per_step_states)
            return summaries, per_step_states, final_states
        return summarize

    def summarize_forward(self, *args, **kwargs):
        summarize = self._summarize_with_fn(self._summarize_forward_fn)
        return summarize(*args, **kwargs)

    def summarize_backward(self, graph, external, sequence,
                           initial_states=None):
        summarize = self._summarize_with_fn(self._summarize_backward_fn)
        noise = tf.random.normal(
            shape=tf.shape(sequence),
            mean=0.0, stddev=self.noise_scale
        )
        sequence = tf.math.add(sequence, noise)
        return summarize(graph, external, sequence, initial_states)

    def extract_summaries(self, rnn_states):
        return extract_rnn_output(rnn_states, self.rnn_num_layers, util.LSTM)

    def refresh_beliefs(self, graph, belief_states, external, observations):
        new_inputs = self._concat_input_and_observ(external, observations)
        return self._refresh_belief_fn(graph, belief_states, new_inputs)

    def perturb(self, observations):
        if not self.denoising:
            return observations
        noise = tf.random.normal(
            shape=tf.shape(observations),
            mean=0.0, stddev=self.noise_scale
        )
        return tf.math.add(observations, noise)

    def concat_conditions(self, external, observations, beliefs, lookaheads):
        if self.use_bi_rnn:
            assert beliefs is not None and lookaheads is not None
            return tf.concat([beliefs, lookaheads], axis=-1)
        elif self.use_belief:
            assert beliefs is not None
            return beliefs
        elif self.use_lookahead:
            assert lookaheads is not None
            return lookaheads
        return self._concat_input_and_observ(external, observations)

    def combine_global(self, inputs, priors, conditions, observations):
        observed_conditions = tf.concat([
            tf.math.reduce_mean(conditions, axis=-2),
            tf.math.reduce_logsumexp(conditions, axis=-2)
        ], axis=-1)
        if inputs is not None:
            observed_conditions = tf.concat(
                [observed_conditions, inputs], axis=-1
            )
        return self._global_skip_cond_gated_unit(
            priors.hidden_conditions, observed_conditions
        )

    def combine_local(self, inputs, priors, conditions, observations):
        observed_conditions = conditions
        if inputs is not None:
            observed_conditions = tf.concat(
                [observed_conditions, inputs], axis=-1
            )
        observed_conditions = tf.concat(
            [observed_conditions, observations], axis=-1
        )
        return self._local_skip_cond_gated_unit(
            priors.hidden_conditions, observed_conditions
        )

    def propose_global(self, graph, external, global_priors,
                       observations, conditions, context=None):
        '''
        Args:
          graph: A RuntimeGraph object.
          observations: A (B, N, dx) Tensor.
          beliefs: A (B, N, 2*dh) Tensor.
          lookaheads: A (B, N, 2*dh) Tensor.
          length: A scalar Tensor, the length of seqeuence.

        Returns:
          proposal: Distribution of batch shape (B, N) and event shape (dz).
        '''
        del context

        global_pre_flow_dist = global_priors.pre_flow_dist

        global_context = self.combine_global(
            inputs=external.global_inputs,
            priors=global_priors,
            conditions=conditions,
            observations=observations
        )

        approx_dist = self._global_mlp_normal(global_context)
        if self.use_skip_conn:
            assert type(approx_dist) is util.PartialLocScaleDist
            approx_dist.loc = self._global_skip_conn_layer_norm(
                tf.math.add(approx_dist.loc, global_pre_flow_dist.mean())
            )

        if self.use_gated_adder:
            assert type(approx_dist) is util.PartialLocScaleDist
            approx_dist.loc = tf.math.add(
                approx_dist.loc,
                self._global_gated_unit(global_context)
            )

        if type(approx_dist) is util.PartialLocScaleDist:
            approx_dist = approx_dist.build()

        if self.global_flow and self.flow_num_layers > 0:
            approx_dist = real_nvp_wrapper(
                components=self._global_flow_components,
                context=global_context,
                base_dist=approx_dist,
                skip_conn=self.flow_skip_conn
            )

        assert approx_dist.reparameterization_type == \
            tfd.FULLY_REPARAMETERIZED
        return approx_dist

    def propose_local(self, graph, external, local_priors,
                      observations, conditions, context=None):
        '''
        Args:
          graph: A RuntimeGraph object.
          histories: A (..., B, N, dh) Tensor.
          states: A (..., B, N, dz) Tensor.
          observations: A (B, N, dx) Tensor.
          beliefs: A (B, N, dh) Tensor.
          lookaheads: A (B, N, dh) Tensor.
          length: A scalar Tensor, the length of seqeuence.

        Returns:
          dist: Distribution of batch shape (..., B, N) & event shape (dz)
        '''
        del context

        local_pre_flow_dist = local_priors.pre_flow_dist

        local_context = self.combine_local(
            inputs=external.local_inputs,
            priors=local_priors,
            conditions=conditions,
            observations=observations
        )
        broadcast_observations = tf.math.add(
            tf.zeros(tf.stack([
                *tf.unstack(tf.shape(local_context)[:-1]),
                util.dim(observations)
            ])),
            observations
        )
        approx_dist = self._local_mlp_normal(
            tf.concat([local_context, broadcast_observations], axis=-1)
        )

        if self.use_skip_conn:
            assert type(approx_dist) is util.PartialLocScaleDist
            approx_dist.loc = self._local_skip_conn_layer_norm(
                tf.math.add(approx_dist.loc, local_pre_flow_dist.mean())
            )

        if self.use_gated_adder:
            assert type(approx_dist) is util.PartialLocScaleDist
            approx_dist.loc = tf.math.add(
                approx_dist.loc,
                self._local_gated_unit(local_context)
            )

        if type(approx_dist) is util.PartialLocScaleDist:
            approx_dist = approx_dist.build()

        approx_dist = tfd.Independent(
            distribution=approx_dist,
            reinterpreted_batch_ndims=1,
            name="indep_" + approx_dist.name
        )
        if self.flow_num_layers > 0:
            approx_dist = perm_equiv_flow_wrapper(
                components=self._local_flow_components,
                graph=graph,
                const_num_nodes=self.model.const_num_nodes,
                global_context=local_priors.global_context,
                local_context=local_context,
                base_dist=approx_dist,
                skip_conn=self.flow_skip_conn
            )
        assert approx_dist.reparameterization_type == \
            tfd.FULLY_REPARAMETERIZED
        return approx_dist


class Proposal(ProposalBuilder):
    def __init__(self, *args,
                 summarize_unit=SUMMARIZER_RGNN, name="Proposal", **kwargs):
        super(Proposal, self).__init__(
            *args, **kwargs,
            summarize_unit=summarize_unit, name=name
        )


class IndepProposal(ProposalBuilder):
    def __init__(self, *args,
                 summarize_unit=SUMMARIZER_LSTM, name="IndepProposal",
                 **kwargs):
        super(IndepProposal, self).__init__(
            *args, **kwargs,
            summarize_unit=summarize_unit, name=name
        )


################################################################
#                      Not maintained.                         #
################################################################


def multistep_preview_wrapper(proposal, previews, weight):
    if type(proposal) is not tfd.MultivariateNormalDiag:
        raise ValueError("Unsupported proposal distribution.")
    mean, stddev = proposal.mean(), proposal.stddev()
    shifted_mean = tf.math.add(
        tf.math.multiply(1.0 - weight, mean),
        tf.math.multiply(weight, previews)
    )
    return tfd.MultivariateNormalDiag(
        loc=shifted_mean, scale_diag=stddev,
        validate_args=True, allow_nan_stats=False,
        name=proposal.name
    )
