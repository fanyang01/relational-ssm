from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import util

import tensorflow as tf
import tensorflow_probability as tfp
import tensorboard as tb

tfd = tfp.distributions


class FlatDistInfo(object):
    def __init__(self, factor_prior_dist):
        self.num_nodes = factor_prior_dist.num_nodes
        self.dim_global_state = factor_prior_dist.dim_global_state
        self.dim_local_state = factor_prior_dist.dim_local_state

    def split_flat_samples(self, flat_samples):
        global_states = flat_samples[..., :self.dim_global_state]
        local_states = tf.reshape(
            flat_samples[..., self.dim_global_state:],
            tf.stack([
                *tf.unstack(tf.shape(flat_samples)[:-1]),
                self.num_nodes, self.dim_local_state
            ])
        )
        global_states.set_shape(
            [None] * (flat_samples.shape.ndims - 1) + [self.dim_global_state]
        )
        local_states.set_shape(
            [None] * flat_samples.shape.ndims + [self.dim_local_state]
        )
        return (global_states, local_states)

    def flatten_samples(self, samples):
        global_states, local_states = samples
        flat_shape = tf.stack([
            *tf.unstack(tf.shape(global_states)[:-1]),
            self.num_nodes * self.dim_local_state
        ])
        return tf.concat([
            global_states, tf.reshape(local_states, flat_shape)
        ], axis=-1)


class FlatJointPrior(FlatDistInfo):
    def __init__(self, factor_prior_dist):
        super(FlatJointPrior, self).__init__(factor_prior_dist)
        self._factor_prior_dist = factor_prior_dist

    def log_prob(self, flat_samples):
        samples = self.split_flat_samples(flat_samples)
        return self._factor_prior_dist.log_prob(samples)


class FlatJointProposal(FlatDistInfo):
    def __init__(self, factor_prior_dist, proposal_sample_fn):
        super(FlatJointProposal, self).__init__(factor_prior_dist)
        self._proposal_sample_fn = proposal_sample_fn

    def sample(self):
        states, _, log_proposal_prob, _ = self._proposal_sample_fn()
        return self.flatten_samples(states), log_proposal_prob

    def log_prob(self, flat_samples):
        samples = self.split_flat_samples(flat_samples)
        return self._factor_proposal_dist.log_prob(samples)


def flat_likelihood_fn_wrapper(factor_prior_dist, make_likelihood):
    flat_dist_info = FlatDistInfo(factor_prior_dist)

    def _fn(flat_samples):
        samples = flat_dist_info.split_flat_samples(flat_samples)
        return make_likelihood(particles=samples)

    return _fn


def LearnableHIS(global_num_dims, local_num_dims,
                 num_steps=10, max_step_size=0.2, mass_scale=1.0,
                 name="LearnableHIS"):
    with tf.variable_scope(name):
        global_step_size_var = tf.get_variable(
            "global_step_size", shape=[global_num_dims],
            dtype=tf.float32, trainable=True,
            initializer=tf.initializers.zeros
        )
        local_step_size_var = tf.get_variable(
            "local_step_size", shape=[local_num_dims],
            dtype=tf.float32, trainable=True,
            initializer=tf.initializers.zeros
        )
        global_step_size = tf.math.multiply(
            max_step_size, tf.math.sigmoid(global_step_size_var)
        )
        local_step_size = tf.math.multiply(
            max_step_size, tf.math.sigmoid(local_step_size_var)
        )
        tb.summary.histogram("HIS/global_step_size", global_step_size)
        tb.summary.histogram("HIS/local_step_size", local_step_size)

        init_inv_temp_var = tf.get_variable(
            "init_inv_temp", shape=[], dtype=tf.float32, trainable=True,
            initializer=tf.initializers.zeros
        )
        init_inv_temp = tf.sigmoid(init_inv_temp_var)
        tb.summary.scalar("HIS/init_inv_temp", init_inv_temp)

    momentum_scale_factor = tf.constant(mass_scale)

    return HamiltonianImportanceSampler(
        global_num_dims=global_num_dims,
        local_num_dims=local_num_dims,
        num_steps=num_steps,
        init_inv_temp=init_inv_temp,
        global_step_size=global_step_size,
        local_step_size=local_step_size,
        momentum_scale_factor=momentum_scale_factor
    )


def HamiltonianImportanceSampler(
        global_num_dims, local_num_dims, num_steps, init_inv_temp,
        global_step_size, local_step_size, momentum_scale_factor):

    def sched_inv_temp(initial_inverse_temp, t, T):
        inv_sqrt = tf.math.divide(1.0, tf.math.sqrt(initial_inverse_temp))
        quad_ratio = tf.math.square(
            tf.math.divide(tf.to_float(t), tf.to_float(T))
        )
        denominator = tf.math.add(
            inv_sqrt, tf.math.multiply(
                tf.math.subtract(1.0, inv_sqrt), quad_ratio
            )
        )
        return tf.math.square(tf.math.divide(1.0, denominator))

    def sample(prior, variational_prior, make_likelihood, observations):
        '''
        Args:
          prior: A distribution with event shape (dz) and compatiable
              batch shape w.r.t. variational_prior
          make_likelihood: A function that returns a distribution with
              batch shape (..., N) and event shape (dx)
          variational_prior: A distribution with batch shape (..., N) and
              event shape (dz)
          observation: A (B, N, dx) Tensor.
          initial_position: Optional. If given, it should a tensor sampled from
              `variational_prior`

        Returns:
          samples: A (..., N, dz) Tensor.
          weights: A (..., N, dz) Tensor.
        '''
        assert observations.shape.ndims == 3
        num_nodes = tf.shape(observations)[1]
        flat_num_dims = global_num_dims + num_nodes * local_num_dims

        step_size = tf.concat([
            global_step_size, tf.tile(local_step_size, [num_nodes])
        ], axis=0)
        half_step_size = tf.math.divide(step_size, 2.0)

        momentum_scale_diag = tf.math.multiply(
            momentum_scale_factor, tf.ones([flat_num_dims])
        )
        momentum_inv_variance = tf.math.divide(
            1.0, tf.square(momentum_scale_diag)
        )

        def cond(t, *unused_args):
            return tf.less(t, num_steps + 1)

        def body(t, position, momentum, inv_temp, *unused_args):
            likelihood = make_likelihood(position)
            neg_log_prob = -tf.math.add(
                prior.log_prob(position), likelihood.log_prob(observations)
            )
            with tf.control_dependencies([
                tf.assert_equal(
                    tf.shape(neg_log_prob), tf.shape(position)[:-1]
                )
            ]):
                gradients = tf.gradients(neg_log_prob, position)
                gradient = gradients[0]
            with tf.control_dependencies([
                tf.assert_equal(tf.shape(gradient), tf.shape(position))
            ]):
                new_momentum_tmp = tf.math.subtract(
                    momentum,
                    tf.math.multiply(half_step_size, gradient)
                )

            new_momentum_tmp_rescaled = tf.math.multiply(
                momentum_inv_variance, new_momentum_tmp
            )
            new_position = tf.math.add(
                position,
                tf.math.multiply(step_size, new_momentum_tmp_rescaled)
            )

            new_likelihood = make_likelihood(new_position)
            new_neg_log_prob = -tf.math.add(
                prior.log_prob(new_position),
                new_likelihood.log_prob(observations)
            )
            new_gradient = tf.gradients(new_neg_log_prob, new_position)[0]
            new_momentum = tf.math.subtract(
                new_momentum_tmp,
                tf.math.multiply(half_step_size, new_gradient)
            )

            new_inv_temp = sched_inv_temp(init_inv_temp, t, num_steps)
            tempering_factor = tf.math.sqrt(
                tf.math.divide(inv_temp, new_inv_temp)
            )
            new_tempered_momentum = tf.math.multiply(
                new_momentum, tempering_factor
            )

            return t + 1, new_position, new_tempered_momentum, new_inv_temp

        initial_position, initial_log_v_prior_prob = \
            variational_prior.sample()
        momentum_loc = tf.zeros_like(initial_position)
        initial_tempered_scale_diag = tf.math.divide(
            momentum_scale_diag, tf.math.sqrt(init_inv_temp)
        )
        initial_momentum_dist = tfd.MultivariateNormalDiag(
            loc=momentum_loc, scale_diag=initial_tempered_scale_diag
        )
        initial_momentum = initial_momentum_dist.sample(1)[0]
        with tf.control_dependencies([
            tf.assert_equal(
                tf.shape(initial_momentum), tf.shape(initial_position)
            )
        ]):
            initial_momentum = tf.identity(initial_momentum)
        t1 = tf.constant(1)

        _, position, momentum, inv_temp = tf.while_loop(
            cond, body, [t1, initial_position, initial_momentum, init_inv_temp]
        )
        #
        # Static loop:
        #
        # t, position, momentum, inv_temp = \
        #     t1, initial_position, initial_momentum, init_inv_temp
        # for _ in range(num_steps):
        #     t, position, momentum, inv_temp = body(
        #         t, position, momentum, inv_temp
        #     )
        #
        final_position, final_momentum, final_inv_temp = \
            position, momentum, inv_temp
        with tf.control_dependencies([
            tf.assert_equal(final_inv_temp, 1.0)
        ]):
            final_position = tf.identity(final_position)

        final_momentum_dist = tfd.MultivariateNormalDiag(
            loc=momentum_loc, scale_diag=momentum_scale_diag
        )
        final_likelihood = make_likelihood(final_position)

        log_jacobian = tf.math.multiply(
            tf.math.divide(util.float(flat_num_dims), 2.0),
            tf.math.log(init_inv_temp)
        )
        log_p = tf.math.add(
            tf.math.add(
                prior.log_prob(final_position),
                final_likelihood.log_prob(observations)
            ),
            final_momentum_dist.log_prob(final_momentum)
        )
        log_q = tf.math.subtract(
            tf.math.add(
                initial_log_v_prior_prob,
                initial_momentum_dist.log_prob(initial_momentum)
            ),
            log_jacobian
        )
        return final_position, tf.math.subtract(log_p, log_q)

    return sample
