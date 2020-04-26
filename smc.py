from __future__ import absolute_import
from __future__ import print_function

import util
import resampling

import tensorflow as tf
import tensorflow_probability as tfp

import functools

tfd = tfp.distributions

MULTINOMIAL_RESAMPLING = "multinomial"
RELAXED_RESAMPLING = "relaxed"

indep_multinomial_resampling = functools.partial(
    resampling.independent,
    base_resampler=resampling.batched_multinomial)
indep_relaxed_resampling = functools.partial(
    resampling.independent,
    base_resampler=resampling.batched_relaxed)


def _possibly_stopped(stop_gradient, x):
    stop = tf.to_float(stop_gradient)
    stopped = tf.stop_gradient(x)
    return util.select(stop, stopped, x)


# TODO

def IndepVSMC(model, proposal,
              resample_criterion=resampling.always_resample,
              resample_impl=MULTINOMIAL_RESAMPLING,
              resample_jointly=True,
              hamiltonian_is=None, implementation=0,
              analytic_kl=True, aux_model=None, pred_resample_init=True,
              parallel_iterations=128, swap_memory=True,
              summary_keys=[tf.GraphKeys.SUMMARIES]):

    def IS(prior, proposal, likelihood_fn, observations, init_particles=None):
        '''
        Args:
          prior: A distribution with event shape (dz) and compatiable
              batch shape w.r.t. proposal
          likelihood_fn: A function that returns a distribution with batch
              shape (..., B, N) and event shape (dx)
          proposal: A distribution with batch shape (..., B, N) and
              event shape (dz)
          observations: A (B, N, dz) Tensor.
          initial_particles: Optional. If given, it should be a tensor sampled
              from `proposal`.

        Returns:
          particles: A (..., B, N, dz) Tensor.
          log_weights: A (..., B, N) Tensor.
        '''
        particles = init_particles
        if particles is None:
            particles = proposal.sample(1)[0]
        likelihood = likelihood_fn(particles)

        log_prior_prob = prior.log_prob(particles)
        log_likelihood_prob = likelihood.log_prob(observations)
        log_proposal_prob = proposal.log_prob(particles)
        log_weights = tf.math.subtract(
            tf.math.add(log_prior_prob, log_likelihood_prob),
            log_proposal_prob
        )
        return particles, log_weights

    def HIS(prior, proposal, likelihood_fn, observations,
            initial_particles=None):
        particles, log_weights = hamiltonian_is(
            prior=prior, make_likelihood=likelihood_fn,
            variational_prior=proposal, observation=observations,
            initial_position=initial_particles
        )
        # TODO: only move masked particles
        return particles, log_weights

    def possibly_centered_sampler_wrapper(independent_sampler):
        def possibly_centered_sampler(
                prior, likelihood_fn, proposal, observations,
                mask, initial_particles=None):
            '''
            Wrap an independent importance sampler to be a masked sampler.

            Args:
              <Same as `IS`>
              mask: A (B, N) Tensor.

            Returns:
              particles: A (S, B, N, dz) Tensor.
              log_weights: A (S, B, N) Tensor.
            '''
            if (not resample_jointly) and (initial_particles is None):
                particles = proposal.sample(1)[0]
                replicated = tf.tile(
                    tf.expand_dims(particles[0], 0),
                    [tf.shape(particles)[0], 1, 1, 1]
                )
                replicated.set_shape(particles.shape)
                mask = tf.expand_dims(mask, -1)  # (B, N, 1)
                initial_particles = util.select(mask, particles, replicated)
            return independent_sampler(
                prior=prior, likelihood_fn=likelihood_fn,
                proposal=proposal, observations=observations,
                initial_particles=initial_particles
            )
        return possibly_centered_sampler

    def possibly_joint_sampler_wrapper(independent_sampler):
        def possibly_joint_sampler(
                prior, likelihood_fn, proposal, observations, mask,
                initial_particles=None):
            '''
            Wrap an independent importance sampler to be a joint sampler.

            Args:
              <Same as `IS`>
              mask: A (B, N) Tensor.

            Returns:
              particles: A (..., B, N, dz) Tensor.
              log_weights: A (..., B) Tensor.
            '''
            particles, log_weights = independent_sampler(
                prior=prior, likelihood_fn=likelihood_fn,
                proposal=proposal, observations=observations,
                initial_particles=initial_particles
            )
            if resample_jointly:
                log_weights = tf.math.multiply(log_weights, mask)
                log_weights = tf.math.reduce_sum(log_weights, axis=-1)

            return particles, log_weights
        return possibly_joint_sampler

    IMPORTANCE_SAMPLER = IS if hamiltonian_is is None else HIS
    IMPORTANCE_SAMPLER = possibly_joint_sampler_wrapper(IMPORTANCE_SAMPLER)

    indep_resamping = indep_multinomial_resampling \
        if resample_impl == MULTINOMIAL_RESAMPLING \
        else indep_relaxed_resampling


def NASMC(model, proposal,
          resample_criterion=always_resample,
          resample_impl=MULTINOMIAL_RESAMPLING,
          hamiltonian_is=None, implementation=0,
          analytic_kl=True, aux_model=None, pred_resample_init=True,
          parallel_iterations=128, swap_memory=True,
          summary_keys=[tf.GraphKeys.SUMMARIES]):

    def init_nasmc_acc(batch_size):
        return tf.zeros([batch_size])

    def update_nasmc_acc(nasmc_acc, new_nasmc_obj, enable=True):
        return util.select(
            enable, tf.math.add(nasmc_acc, new_nasmc_obj), nasmc_acc
        )

    def local_nasmc_objective(
            ancestor_particles, ancestor_histories,
            make_prior, make_likelihood, make_proposal,
            particles, log_weights, observations):
        stopped_ancestor_particles = tf.stop_gradient(ancestor_particles)
        stopped_ancestor_histories = tf.stop_gradient(ancestor_histories)
        priors = make_prior(
            histories=stopped_ancestor_histories,
            states=stopped_ancestor_particles
        )
        prior_dist = priors.next_state_dist
        histories = priors.refreshed_histories
        proposal_dist = make_proposal(
            histories=stopped_ancestor_histories,
            states=stopped_ancestor_particles,
            priors=priors
        )
        stopped_particles = tf.stop_gradient(particles)
        stopped_histories = tf.stop_gradient(histories)
        likelihood = make_likelihood(
            histories=stopped_histories, states=stopped_particles
        )
        stopped_log_weights = tf.stop_gradient(log_weights)
        normalized_weights = tf.math.softmax(stopped_log_weights, axis=0)
        expectation_inner_part = tf.math.reduce_sum(tf.math.add_n([
            prior_dist.log_prob(stopped_particles),
            likelihood.log_prob(observations),
            proposal_dist.log_prob(stopped_particles)
        ]), axis=-1)  # (S, B, N) -> (S, B)
        return tf.math.reduce_sum(
            tf.math.multiply(normalized_weights, expectation_inner_part),
            axis=0
        )  # (B)
