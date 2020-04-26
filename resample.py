from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import util

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


# TODO: add support for mask.

def ess_criterion(log_ess, num_samples):
    return tf.less(log_ess, tf.log(num_samples / 2.0))


def always_resample(log_ess, *used_args):
    return tf.fill(tf.shape(log_ess), True)


def never_resample(log_ess, *used_args):
    return tf.fill(tf.shape(log_ess), False)


def batched_gather(particles, ancestor_indices):
    '''
    Args:
      particles: A (S, B, ...) Tensor.
      ancestor_indices: A (B, NS) Tensor.

    Returns:
      resampled_particles: A (NS, B, ...) Tensor.
    '''
    ancestor_indices = tf.transpose(ancestor_indices)  # (NS, B)
    num_samples = tf.shape(ancestor_indices)[0]
    shape = tf.shape(particles)
    num_particles, batch_size = shape[0], shape[1]
    particles_main_shape_list = tf.unstack(shape[2:])
    #
    # Proof-of-Concept Implementation:
    #
    # batch_indices = tf.tile(
    #     tf.expand_dims(tf.arange(batch_size), 0),
    #     tf.stack([num_samples, 1])
    # )  # (B) -> (1, B) -> (NS, B)

    # indices_2d = tf.concat(
    #     [
    #         tf.expand_dims(ancestor_indices, -1),
    #         tf.expand_dims(batch_indices, -1)
    #     ], axis=-1
    # )  # (NS, B) -> (NS, B, 1) -> (NS, B, 2)

    # # (S, B, ...) --> gather_nd --> (NS, B, ...)
    # resampled_particles = tf.gather_nd(particles, indices_2d)
    #
    # Faster Implementation:
    #
    offset = tf.expand_dims(tf.range(batch_size), 0)
    # row_idx * num_col + col_idx to index into a flattened (S * B) array
    flat_indices = tf.reshape(
        tf.math.add(tf.math.multiply(ancestor_indices, batch_size), offset),
        [num_samples * batch_size]  # (NS, B) -> (NS * B)
    )
    flat_particles = tf.gather(
        tf.reshape(particles, tf.stack([
            num_particles * batch_size, *particles_main_shape_list
        ])),
        flat_indices
    )
    resampled_particles = tf.reshape(
        flat_particles, [num_samples, batch_size, *particles_main_shape_list]
    )
    return resampled_particles


def batched_gather_new(particles, ancestor_indices):
    '''
    Use the new(?) `tf.batch_gather` API.

    Args:
      particles: A (S, B, ...) Tensor.
      ancestor_indices: A (B, NS) Tensor.

    Returns:
      resampled_particles: A (NS, B, ...) Tensor.
    '''
    perm = tf.range(particles.shape.ndims)
    perm = tf.stack([
        perm[1], perm[0], *tf.unstack(perm[2:])
    ])

    # (S, B, ...) -> (B, S, ...) -> (B, NS, ...) -> (NS, B, ...)
    assert ancestor_indices.shape.ndims == 2
    with tf.control_dependencies([tf.assert_equal(
        tf.shape(particles)[1], tf.shape(ancestor_indices)[0]
    )]):
        resampled_particles = tf.batch_gather(
            tf.transpose(particles, perm), ancestor_indices
        )
    return tf.transpose(resampled_particles, perm)


def batched_multinomial(particles, log_weights, num_samples):
    '''
    Args:
      particles: A (S, B, ...) Tensor.
      log_weights: A (S, B) Tensor.
      num_samples: Number of samples to resample.

    Returns:
      resampled_particles: A (NS, B, ...) Tensor.
    '''
    assert log_weights.shape.ndims == 2
    with tf.control_dependencies([tf.assert_equal(
        tf.shape(log_weights), tf.shape(particles)[:2]
    )]):
        logits = tf.transpose(log_weights)  # (S, B) -> (B, S)

    ancestor_indices = tf.stop_gradient(
        tf.random.categorical(logits=logits, num_samples=num_samples)
    )  # (B, NS)
    assert ancestor_indices.shape.ndims == 2

    resampled_particles = batched_gather_new(particles, ancestor_indices)
    resampled_particles.set_shape([
        num_samples, *particles.shape.as_list()[1:]
    ])
    return resampled_particles


def batched_relaxed(particles, log_weights, num_samples, temperature=0.5):
    '''
    Args:
      particles: A (S, B, ...) Tensor.
      log_weights: A (S, B) Tensor.
      num_samples: Number of samples to resample.

    Returns:
      resampled_particles: A (NS, B, ...) Tensor.
    '''
    assert len(log_weights.shape.as_list()) == 2
    shape = tf.shape(log_weights)
    with tf.control_dependencies([
        tf.assert_equal(shape, tf.shape(particles)[:2])
    ]):
        num_particles, batch_size = shape[0], shape[1]
        particles_main_shape_list = tf.unstack(tf.shape(particles)[2:])

    dist = tfd.RelaxedOneHotCategorical(
        temperature, logits=tf.transpose(log_weights, perm=[1, 0])
    )  # (S, B) -> (B, S)
    blending_weights = dist.sample(num_samples)  # (NS, B, S)

    # (S, B, ...) -> (S, B, d) -> (B, d, S)
    particles_T = tf.transpose(
        tf.reshape(particles, tf.stack([num_particles, batch_size, -1])),
        perm=[1, 2, 0]
    )
    # (NS, B, S) -> (B, S, NS)
    blending_weights_T = tf.transpose(blending_weights, perm=[1, 2, 0])
    # (B, d, S) * (B, S, NS) -> (B, d, NS) -> (NS, B, d)
    resampled_particles = tf.transpose(
        tf.linalg.matmul(particles_T, blending_weights_T),
        perm=[2, 0, 1]
    )
    # (NS, B, d) -> (NS, B, ...)
    resampled_particles = tf.reshape(resampled_particles, tf.stack([
        num_samples, batch_size, *particles_main_shape_list
    ]))
    resampled_particles.set_shape([
        num_samples, *particles.shape.as_list()[1:]
    ])

    return resampled_particles


def resampling(resampler, histories, particles, log_weights, num_samples,
               mask=None):
    '''
    Args:
      histories: A 2-ary tuple:
        - global_histories: A (S, B, dH) Tensor.
        - local_histories: A (S, B, N, dH) Tensor.
      particles: A 2-ary tuple:
        - global_states: A (S, B, dz) Tensor.
        - local_states: A (S, B, N, dz) Tensor.
      log_weights: A (S, B) Tensor.
      num_samples: A scalar.

    Returns:
      resampled_histories, resampled_particles, resampled_log_weights
    '''
    global_histories, local_histories = histories
    global_states, local_states = particles
    assert global_states.shape.ndims == 3
    assert local_states.shape.ndims == 4
    shape = tf.shape(local_states)
    prev_num_samples, batch_size, num_nodes = shape[0], shape[1], shape[2]
    dim_global_state = util.dim(global_states)
    dim_local_state = util.dim(local_states)
    dim_global_history = util.dim(global_histories)
    dim_local_history = util.dim(local_histories)

    local_flat_shape = tf.stack([prev_num_samples, batch_size, -1])
    flat_local_states = tf.reshape(local_states, local_flat_shape)
    flat_local_histories = tf.reshape(local_histories, local_flat_shape)

    concated = tf.concat([
        tf.expand_dims(log_weights, axis=-1),
        global_histories, global_states,
        flat_local_histories, flat_local_states
    ], axis=-1)
    resampled = resampler(concated, log_weights, num_samples)

    split_sizes = tf.stack([
        1, dim_global_history, dim_global_state,
        num_nodes * dim_local_history, num_nodes * dim_local_state
    ])
    resampled_log_weights, \
        resampled_global_histories, resampled_global_states, \
        resampled_flat_local_histories, resampled_flat_local_states \
        = tf.split(resampled, split_sizes, axis=-1)

    resampled_local_histories = tf.reshape(
        resampled_flat_local_histories,
        [num_samples, batch_size, num_nodes, dim_local_history]
    )
    resampled_local_states = tf.reshape(
        resampled_flat_local_states,
        [num_samples, batch_size, num_nodes, dim_local_state]
    )

    resampled_histories = (
        resampled_global_histories, resampled_local_histories
    )
    resampled_particles = (
        resampled_global_states, resampled_local_states
    )
    return (
        resampled_histories, resampled_particles,
        tf.squeeze(resampled_log_weights, axis=[-1])
    )


# TODO

def independent(particles, log_weights, num_samples,
                base_resampler=batched_multinomial):
    '''
    Args:
      particles: A (S, B, N, ...) Tensor.
      log_weights: A (S, B, N) Tensor.
      num_samples: Number of samples to resample.

    Returns:
      resampled_particles: A (NS, B, N, ...) Tensor.
    '''
    shape = tf.shape(log_weights)
    with tf.control_dependencies([
        tf.assert_equal(shape, tf.shape(particles)[:3])
    ]):
        num_particles, batch_size, num_nodes = shape[0], shape[1], shape[2]
        particles_main_shape_list = tf.unstack(tf.shape(particles)[3:])

    # (S, B, N, ...) -> (S, B * N, ...)
    flat_particles = tf.reshape(particles, tf.stack([
        num_particles, batch_size * num_nodes, *particles_main_shape_list
    ]))
    flat_log_weights = tf.reshape(log_weights, tf.stack([
        num_particles, batch_size * num_nodes
    ]))
    # (NS, B * N, ...)
    flat_resampled_particles = base_resampler(
        particles=flat_particles, log_weights=flat_log_weights,
        num_samples=num_samples
    )
    # (NS, B, N, ...)
    resampled_particles = tf.reshape(flat_resampled_particles, tf.stack([
        num_samples, batch_size, num_nodes, *particles_main_shape_list
    ]))
    resampled_particles.set_shape([num_samples, *particles.shape[1:]])

    return resampled_particles
