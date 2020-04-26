from __future__ import absolute_import
from __future__ import print_function

import util
import resample
from auxiliary import compute_aux_scores
from proposal import collect_persistent_states
from proposal import ProposalContext
from predict import make_predict_fn
from hvae import FlatJointPrior, FlatJointProposal
from hvae import flat_likelihood_fn_wrapper

import tensorflow as tf
import tensorflow_probability as tfp

import functools

tfd = tfp.distributions

MULTINOMIAL_RESAMPLING = "multinomial"
RELAXED_RESAMPLING = "relaxed"


class Estimation(object):
    def __init__(self, vsmc_bound, aux_score,
                 avg_ess, avg_kl, avg_resample):
        self.vsmc_bound = vsmc_bound
        self.aux_score = aux_score
        self.avg_resample = avg_resample
        self.avg_ess = avg_ess
        self.avg_kl = avg_kl

    def install_summaries(self, collection, family):
        summ_scalar = functools.partial(
            tf.summary.scalar,
            collections=collection, family=family
        )
        summ_scalar("vsmc_bound", self.vsmc_bound)
        summ_scalar("aux_score", self.aux_score)
        summ_scalar("avg_resample", self.avg_resample)
        summ_scalar("avg_ess", self.avg_ess)
        summ_scalar("avg_kl", self.avg_kl)


joint_multinomial_resampling = resample.batched_multinomial
joint_relaxed_resampling = resample.batched_relaxed


def VSMC(model, proposal,
         interleaving_scheduler=None,
         resample_criterion=resample.always_resample,
         resample_impl=MULTINOMIAL_RESAMPLING,
         hamiltonian_is=None, implementation=0,
         analytic_kl=True, aux_model=None, pred_resample_init=True,
         parallel_iterations=10, swap_memory=False,
         summary_keys=[tf.GraphKeys.SUMMARIES]):

    assert (
        (resample_impl == MULTINOMIAL_RESAMPLING) or
        (resample_impl == RELAXED_RESAMPLING)
    )
    joint_resampling = joint_multinomial_resampling \
        if resample_impl == MULTINOMIAL_RESAMPLING \
        else joint_relaxed_resampling
    RESAMPLER = functools.partial(resample.resampling, joint_resampling)

    def LOG_WEIGHT(log_prior_prob, log_proposal_prob, log_likelihood_prob):
        return tf.math.subtract(
            tf.math.add(log_prior_prob, log_likelihood_prob),
            log_proposal_prob
        )

    def IS(prior_dist, proposal_fn, likelihood_fn, observations):
        particles, histories, log_proposal_prob, log_prior_prob = \
            proposal_fn()

        likelihood = likelihood_fn(histories=histories, particles=particles)
        log_likelihood_prob = likelihood.log_prob(observations)

        log_weights = LOG_WEIGHT(
            log_prior_prob=log_prior_prob,
            log_proposal_prob=log_proposal_prob,
            log_likelihood_prob=log_likelihood_prob
        )
        kl = tf.math.subtract(log_proposal_prob, log_prior_prob)
        return histories, particles, log_weights, kl

    def HIS(prior_dist, proposal_fn, likelihood_fn, observations):
        flat_prior_dist = FlatJointPrior(prior_dist)
        flat_proposal_dist = FlatJointProposal(prior_dist, proposal_fn)
        likelihood_fn = functools.partial(likelihood_fn, histories=None)
        likelihood_fn = flat_likelihood_fn_wrapper(prior_dist, likelihood_fn)

        flat_particles, log_weights = hamiltonian_is(
            prior=flat_prior_dist,
            variational_prior=flat_proposal_dist,
            make_likelihood=likelihood_fn,
            observations=observations
        )
        particles = flat_prior_dist.split_flat_samples(flat_particles)
        return None, particles, log_weights, tf.zeros_like(log_weights)

    def make_likelihood_quick(graph, external, global_priors,
                              histories, particles):
        del graph, external, global_priors

        indep_dists = model.emit(histories=histories, states=particles)
        assert indep_dists.event_shape.ndims == 1
        return tfd.Independent(
            distribution=indep_dists,
            reinterpreted_batch_ndims=1,
            name="indep_" + indep_dists.name
        )

    def make_likelihood_slow(graph, external, global_priors,
                             histories, particles):
        del histories  # In HIS, `histories` should be refreshed.

        global_states, _ = particles
        local_priors = model.trans_local(
            graph, external, global_priors, global_states
        )
        histories = local_priors.full_histories
        return make_likelihood_quick(
            graph, external, global_priors, histories, particles
        )

    if hamiltonian_is is not None:
        IMPORTANCE_SAMPLER = HIS
        MAKE_LIKELIHOOD = make_likelihood_slow
    else:
        IMPORTANCE_SAMPLER = IS
        MAKE_LIKELIHOOD = make_likelihood_quick

    def init_particles(
            graph, external, observations, conditions, num_particles,
            length, initial_states=None, initial_histories=None):
        assert (initial_states is None) == (initial_histories is None)

        context = ProposalContext(0, length)
        if initial_states is None:
            global_priors = model.init_global(
                graph=graph, external=external,
                num_samples=num_particles,
                prefix_shape=tf.shape(observations)[:-1]
            )
        else:
            global_priors = model.trans_global(
                graph=graph, external=external,
                histories=initial_histories, states=initial_states,
                observations=tf.zeros_like(observations)  # TODO
            )
        factor_prior_dist = model.factorized_prior(
            graph=graph, external=external, global_priors=global_priors
        )
        factor_proposal_dist = proposal.propose(
            graph=graph, external=external, global_priors=global_priors,
            observations=observations, conditions=conditions,
            context=context
        )
        likelihood_fn = functools.partial(
            MAKE_LIKELIHOOD,
            graph=graph, external=external, global_priors=global_priors
        )
        histories, particles, log_weights, _ = IMPORTANCE_SAMPLER(
            prior_dist=factor_prior_dist,
            proposal_fn=factor_proposal_dist.sample,
            likelihood_fn=likelihood_fn,
            observations=observations
        )
        histories = model.recover_histories_if_stale(
            graph, external, global_priors, histories, particles
        )
        return histories, particles, log_weights

    def summ_batch_stats(graph, x, avg=False):
        '''
        Args:
          x: A (B) Tensor.

        Returns:
          avg: A scalar Tensor.
        '''
        x = graph.batch_avg(x) if avg else x
        return tf.math.reduce_mean(x)

    def estimate_log_ess(log_weights):
        '''
        Args:
          log_weights: A (S, ...) Tensor.

        Returns:
          log_ess: A (...) Tensor.
        '''
        # Effective sample size: ESS = sum(w_i)^2 / sum(w_i^2)
        log_num = tf.multiply(2.0, tf.reduce_logsumexp(log_weights, axis=0))
        log_denum = tf.reduce_logsumexp(tf.multiply(log_weights, 2.0), axis=0)
        return tf.subtract(log_num, log_denum)

    def cond_resample(t, histories, particles, log_weights,
                      const_num_particles, mask):
        '''
        Args:
          histories: A 2-ary tuple of Tensors: (S, B, dH) and (S, B, N, dH).
          particles: A 2-ary tuple of Tensors: (S, B, dz) and (S, B, N, dz).
          log_weights: A (S, B) Tensor.
          const_num_particles: S.
          mask: A (B, N) Tensor, 0: don't resample; 1: resample.

        Returns:
          ancestral_particles: A 2-ary tuple of Tensors.
          ancestral_histories: A 2-ary tuple of Tensors.
          log_ess: A (B) Tensor.
          resampled: A (B) Tensor.
        '''
        resampled_histories, resampled_particles, \
            resampled_log_weights = RESAMPLER(
                histories=histories, particles=particles,
                log_weights=log_weights, num_samples=const_num_particles
            )

        log_ess = estimate_log_ess(log_weights)  # (B)
        should_resample = resample_criterion(log_ess, const_num_particles)
        resample_mask = tf.expand_dims(util.float(should_resample), axis=0)

        # (1, B, ...) * (S, B, ...)
        ancestral_log_weights, \
            ancestral_histories, ancestral_particles = util.select_nested(
                resample_mask,
                (
                    resampled_log_weights,
                    resampled_histories, resampled_particles
                ), (
                    log_weights,
                    histories, particles
                ),
                expand=True, set_shape=True
            )
        resampled = should_resample

        return (
            ancestral_histories, ancestral_particles,
            ancestral_log_weights, log_ess, resampled
        )

    def cond_propose(context, graph, external, batch_size,
                     global_priors, factor_proposal_dist):
        proposed_particles, proposed_histories, \
            proposed_log_proposal_prob, proposed_log_prior_prob = \
            factor_proposal_dist.sample()

        prior_particles, prior_histories, prior_log_prob = \
            model.sample_latent_states(graph, external, global_priors)
        prior_log_proposal_prob = prior_log_prior_prob = prior_log_prob

        # (B) -> (1, B)
        interleaving_mask = tf.ones([batch_size]) \
            if interleaving_scheduler is None \
            else interleaving_scheduler.sched(context, batch_size)
        interleaving_mask = tf.expand_dims(interleaving_mask, axis=0)

        return util.select_nested(
            interleaving_mask,
            (
                proposed_particles, proposed_histories,
                proposed_log_proposal_prob, proposed_log_prior_prob
            ), (
                prior_particles, prior_histories,
                prior_log_proposal_prob, prior_log_prior_prob
            ),
            expand=True, set_shape=True
        )

    def forward(t, graph, external, num_particles,
                histories, particles, log_weights,
                old_observations, new_observations,
                conditions, length):
        '''
        Args:
          graph: A graph.RuntimeGraph object.
          num_particles: Number of new particles.
          particles: A 2-ary tuple:
            - global_states: A (S, B, dz) Tensor.
            - local_states: A (S, B, N, dz) Tensor.
          histories: A 2-ary tuple:
            - global_histories: A (S, B, dH) Tensor.
            - local_histories: A (S, B, N, dH) Tensor.
          log_weights: A (S, B) Tensor.
          inputs: A (B,[ N,] dx) Tensor or None.
          observations: A (B, N, dx) Tensor.
          next_observations: A (B, N, dx) Tensor.
          conditions: A (B, N, dh) Tensor.

        Returns:
          new_particles: A (S, B, N, dz) Tensor.
          new_log_alphas: A (S, B) Tensor.
          new_histories: A (S, B, N, dh) Tensor.
          resampled: A (B) Tensor, whether the filters are resampled.
          avg_ess: The effective sample size.
          avg_kl: Averaged KL(proposal||transition).
        '''
        global_histories, local_histories = histories
        global_particles, local_particles = particles

        shape = tf.shape(local_particles)
        with tf.control_dependencies([
            tf.assert_equal(tf.size(shape), 4),
            tf.assert_equal(tf.shape(new_observations)[:-1], shape[1:-1]),
            tf.assert_equal(num_particles, shape[0]),
            tf.assert_equal(num_particles, tf.shape(log_weights)[0])
        ]):
            batch_size = tf.shape(new_observations)[0]

        ancestor_histories, ancestor_particles, \
            ancestor_log_weights, log_ess, resampled = cond_resample(
                t=t, histories=histories, particles=particles,
                log_weights=log_weights, const_num_particles=num_particles,
                mask=graph.node_mask
            )

        context = ProposalContext(t, length)
        global_priors = model.trans_global(
            graph=graph, external=external,
            histories=ancestor_histories,
            states=ancestor_particles,
            observations=old_observations
        )
        factor_prior_dist = model.factorized_prior(
            graph=graph, external=external,
            global_priors=global_priors
        )
        factor_proposal_dist = proposal.propose(
            graph=graph,
            external=external,
            global_priors=global_priors,
            observations=new_observations,
            conditions=conditions,
            context=context
        )
        proposal_fn = functools.partial(
            cond_propose,
            context=context, graph=graph, external=external,
            batch_size=batch_size,
            global_priors=global_priors,
            factor_proposal_dist=factor_proposal_dist
        )
        likelihood_fn = functools.partial(
            MAKE_LIKELIHOOD,
            graph=graph, external=external, global_priors=global_priors
        )

        new_histories, new_particles, new_log_alphas, kl = IMPORTANCE_SAMPLER(
            prior_dist=factor_prior_dist,
            proposal_fn=proposal_fn,
            likelihood_fn=likelihood_fn,
            observations=new_observations
        )
        new_histories = model.recover_histories_if_stale(
            graph, external, global_priors, new_histories, new_particles
        )

        new_histories = util.nested_set_shape_like(new_histories, histories)
        new_particles = util.nested_set_shape_like(new_particles, particles)

        # Estimate KL(proposal||transition) for debugging.
        # (S, B) -> scalar
        avg_kl = tf.math.reduce_mean(graph.batch_avg(kl))

        # (B) -> scalar
        avg_ess = summ_batch_stats(graph, tf.math.exp(log_ess))
        count = summ_batch_stats(graph, util.float(resampled))

        return (
            new_histories, new_particles, new_log_alphas,
            resampled, count, avg_ess, avg_kl
        )

    def init_accs():
        return (0.0, 0.0, 0.0)

    def update_accs(accumulators, resampled, ess, kl, mask):
        resample_acc, ess_acc, kl_acc = accumulators
        resampled = util.float(resampled)
        resample_acc += tf.math.reduce_mean(resampled)
        ess_acc += ess
        kl_acc += kl
        new_accumulators = (resample_acc, ess_acc, kl_acc)
        return new_accumulators

    def finalize_accs(accumulators, num_time_steps):
        resample_acc, ess_acc, kl_acc = accumulators
        num_time_steps = util.float(num_time_steps)
        avg_resample = tf.math.divide(resample_acc, num_time_steps)
        avg_ess = tf.math.divide(ess_acc, num_time_steps)
        avg_kl = tf.math.divide(kl_acc, num_time_steps)
        return avg_resample, avg_ess, avg_kl

    def init_log_Z_est_0(initial_log_weights):
        initial_log_Z_est_acc = tf.zeros(tf.shape(initial_log_weights)[1:])
        initial_log_Z_est_prev = tf.math.subtract(
            tf.math.reduce_logsumexp(initial_log_weights, axis=0),
            tf.math.log(util.float(tf.shape(initial_log_weights)[0]))
        )  # (S, B) -> (B)
        initial_log_Z_est = (initial_log_Z_est_acc, initial_log_Z_est_prev)
        return initial_log_Z_est

    def update_log_Z_est_0(
            log_Z_est, log_weights,
            new_log_alphas, new_log_weights, resampled, enable=True):
        del log_weights, new_log_alphas

        log_Z_est_acc, log_Z_est_prev = log_Z_est
        new_log_Z_est_acc = tf.math.add(
            log_Z_est_acc,
            tf.math.multiply(log_Z_est_prev, util.float(resampled))
        )
        new_log_Z_est_prev = tf.math.subtract(
            tf.math.reduce_logsumexp(new_log_weights, axis=0),
            tf.math.log(util.float(tf.shape(new_log_weights)[0]))
        )

        new_log_Z_est_acc = util.select(
            enable, new_log_Z_est_acc, log_Z_est_acc)
        new_log_Z_est_prev = util.select(
            enable, new_log_Z_est_prev, log_Z_est_prev)
        new_log_Z_est = (new_log_Z_est_acc, new_log_Z_est_prev)

        return new_log_Z_est, new_log_Z_est_acc, new_log_Z_est_prev

    def finalize_log_Z_est_0(log_Z_est):
        log_Z_est_acc, log_Z_est_prev = log_Z_est
        return tf.math.add(log_Z_est_acc, log_Z_est_prev)

    def init_log_Z_est_1(initial_log_weights):
        initial_log_Z_est = tf.math.subtract(
            tf.math.reduce_logsumexp(initial_log_weights, axis=0),
            tf.math.log(util.float(tf.shape(initial_log_weights)[0]))
        )
        return initial_log_Z_est

    def update_log_Z_est_1(
            log_Z_est, log_weights,
            new_log_alphas, new_log_weights, resampled, enable=True):
        del new_log_weights

        new_log_Z_est_resampled = tf.math.subtract(
            tf.math.reduce_logsumexp(new_log_alphas, axis=0),
            tf.math.log(util.float(tf.shape(new_log_alphas)[0]))
        )
        new_log_Z_est_not_resampled = tf.math.reduce_logsumexp(
            tf.math.add(
                tf.math.log(tf.math.softmax(log_weights, axis=0)),
                new_log_alphas
            ), axis=0
        )
        new_log_Z_est_update = util.select(
            resampled,
            new_log_Z_est_resampled, new_log_Z_est_not_resampled
        )
        new_log_Z_est = tf.math.add(log_Z_est, new_log_Z_est_update)
        new_log_Z_est = util.select(enable, new_log_Z_est, log_Z_est)
        return new_log_Z_est, new_log_Z_est, new_log_Z_est_update

    def finalize_log_Z_est_1(log_Z_est):
        return log_Z_est

    def update_log_weights(log_weights, new_log_alphas, resampled):
        log_weights_acc = tf.math.multiply(
            log_weights, tf.math.subtract(1.0, util.float(resampled))
        )
        return tf.math.add(log_weights_acc, new_log_alphas)

    def estimate(
            mode, graph, external, observations, num_particles,
            initial_belief_states=None,
            initial_latent_histories=None,
            initial_latent_states=None):
        '''
        Args:
          mode: A string tensor, 'TRAIN' or 'EVAL'.
          graph: A RuntimeGraph object.
          observations: A (T, B, N, dx) Tensor.
          num_particles: Number of particles to use.
          initial_belief_states: Optional. A (B, N, dh) Tensor.
          initial_belief_states: Optional. A (S, B, N, dz) Tensor.
          initial_latent_histories: Optional. A (S, B, N, dh) Tensor.

        Returns:
          log_Z_est: The unbiased likelihood estimator of
            p(y_{1:T}) = p(y_1)p(y_2|y_1)...p(y_T|y_{1:T-1}).
          last_belief_states: A (B, N, dh) Tensor.
          last_latent_states: A (S, B, N, dz) Tensor.
          last_latent_histories: A (S, B, N, dh) Tensor.
        '''
        assert observations.shape.ndims == 4
        num_time_steps = tf.shape(observations)[0]

        perturbed_observations = proposal.perturb(observations)

        beliefs, _, last_belief_states = proposal.summarize_forward(
            graph=graph, external=external,
            sequence=perturbed_observations,
            initial_states=initial_belief_states
        )
        lookaheads, _, _ = proposal.summarize_backward(
            graph=graph, external=external,
            sequence=perturbed_observations,
            initial_states=last_belief_states
        )
        lookaheads = util.left_shift_and_pad(
            lookaheads,
            proposal.extract_summaries(last_belief_states)
        )
        conditions = proposal.concat_conditions(
            external=external, observations=perturbed_observations,
            beliefs=beliefs, lookaheads=lookaheads
        )

        impl = implementation
        impls = [
            (init_log_Z_est_0, update_log_Z_est_0, finalize_log_Z_est_0),
            (init_log_Z_est_1, update_log_Z_est_1, finalize_log_Z_est_1)
        ]
        init_log_Z_est, update_log_Z_est, finalize_log_Z_est = impls[impl]

        def cond(t, *unused_args):
            return tf.less(t, num_time_steps)

        def body(t, histories, particles, log_weights,
                 log_Z_est, aux_scores, accumulators):
            new_histories, new_particles, new_log_alphas, \
                resampled, count, ess, kl = forward(
                    t=t, graph=graph, external=external.current(t),
                    num_particles=num_particles,
                    histories=histories, particles=particles,
                    log_weights=log_weights,
                    old_observations=perturbed_observations[t - 1],
                    new_observations=observations[t],
                    conditions=conditions[t],
                    length=num_time_steps
                )
            new_log_weights = update_log_weights(
                log_weights, new_log_alphas, resampled
            )
            new_log_Z_est, new_log_Z_acc, new_log_Z_t = update_log_Z_est(
                log_Z_est=log_Z_est, log_weights=log_weights,
                new_log_alphas=new_log_alphas,
                new_log_weights=new_log_weights,
                resampled=resampled, enable=True
            )

            unweighted_histories, unweighted_particles, _ = RESAMPLER(
                histories=new_histories, particles=new_particles,
                log_weights=new_log_weights, num_samples=num_particles
            )
            aux_scores_t = compute_aux_scores(
                aux_model, graph,
                model.extract_all_rnn_output(unweighted_histories),
                unweighted_particles,
                observations[t:], beliefs[t], lookaheads[t]
            )
            new_aux_scores = tf.math.add(aux_scores, aux_scores_t)

            new_accumulators = update_accs(
                accumulators, resampled, ess, kl, mask=graph.node_mask
            )

            print_op = tf.print(
                "[mode step t res ess kl logZt logZ aux] = ",
                mode, tf.train.get_global_step(), t,
                count, ess, kl,
                summ_batch_stats(graph, new_log_Z_t),
                summ_batch_stats(graph, new_log_Z_acc),
                tf.math.reduce_mean(aux_scores_t)
            )
            with tf.control_dependencies([print_op]):
                new_t = t + 1

            return (
                new_t, new_histories, new_particles, new_log_weights,
                new_log_Z_est, new_aux_scores, new_accumulators
            )

        t1 = tf.constant(1)
        initial_histories, initial_particles, initial_log_weights = \
            init_particles(
                graph=graph,
                external=external.current(0),
                observations=observations[0],
                conditions=conditions[0],
                num_particles=num_particles,
                length=num_time_steps,
                initial_histories=initial_latent_histories,
                initial_states=initial_latent_states
            )
        initial_log_Z_est = init_log_Z_est(initial_log_weights)
        initial_accumulators = init_accs()

        unweighted_histories, unweighted_particles, _ = RESAMPLER(
            histories=initial_histories, particles=initial_particles,
            log_weights=initial_log_weights, num_samples=num_particles
        )
        initial_aux_scores = compute_aux_scores(
            aux_model, graph,
            model.extract_all_rnn_output(unweighted_histories),
            unweighted_particles,
            observations[0:], beliefs[0], lookaheads[0]
        )

        T, last_histories, last_particles, last_log_weights, \
            log_Z_est, aux_scores, accumulators = tf.while_loop(
                cond, body,
                [
                    t1, initial_histories, initial_particles,
                    initial_log_weights,
                    initial_log_Z_est,
                    initial_aux_scores,
                    initial_accumulators
                ],
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory
            )

        log_Z_est = finalize_log_Z_est(log_Z_est)
        assert log_Z_est.shape.ndims == 1  # (B)
        log_Z_est = tf.math.reduce_mean(log_Z_est)

        assert aux_scores.shape.ndims == 2
        avg_aux_score = tf.math.reduce_mean(aux_scores)  # (S, B) -> scalar
        avg_resample, avg_ess, avg_kl = finalize_accs(accumulators, T)

        last_latent_histories, last_latent_states, _ = RESAMPLER(
            histories=last_histories, particles=last_particles,
            log_weights=last_log_weights, num_samples=num_particles
        )

        print_op = tf.print(
            "SUMMARY: [mode step t res ess kl logZ aux] = ",
            mode, tf.train.get_global_step(), T,
            avg_resample, avg_ess, avg_kl,
            log_Z_est, avg_aux_score
        )
        with tf.control_dependencies([print_op]):
            log_Z_est = tf.identity(log_Z_est)

        est = Estimation(
            vsmc_bound=log_Z_est, aux_score=avg_aux_score,
            avg_resample=avg_resample, avg_ess=avg_ess, avg_kl=avg_kl
        )
        states = collect_persistent_states(
            belief_states=last_belief_states,
            latent_histories=last_latent_histories,
            latent_states=last_latent_states
        )
        return est, states

    def init_state_fn(mode, *args, num_samples=None, **kwargs):
        return init_particles(*args, num_particles=num_samples, **kwargs)

    def update_state_fn(mode, t, graph, external, num_samples,
                        histories, states, log_weights, **kwargs):
        new_histories, new_particles, new_log_alphas, \
            resampled, count, ess, kl = forward(
                t=t, graph=graph, external=external,
                num_particles=num_samples,
                histories=histories, particles=states,
                log_weights=log_weights, **kwargs
            )
        new_log_weights = update_log_weights(
            log_weights, new_log_alphas, resampled
        )
        new_log_Z_t = tf.math.subtract(
            tf.math.reduce_logsumexp(new_log_weights, axis=0),
            tf.math.log(util.float(tf.shape(new_log_weights)[0]))
        )
        print_op = tf.print(
            "[mode t resample ess kl logZt] = ",
            mode, t, count, ess, kl,
            summ_batch_stats(graph, new_log_Z_t)
        )
        with tf.control_dependencies([print_op]):
            new_log_weights = tf.identity(new_log_weights)
        return new_histories, new_particles, new_log_weights

    predict = make_predict_fn(
        model=model, proposal=proposal,
        init_state_fn=init_state_fn, update_state_fn=update_state_fn,
        parallel_iterations=parallel_iterations, swap_memory=swap_memory
    )

    return estimate, predict
