from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import generative
import resample
import util

import functools
import tensorflow as tf


RESAMPLING = functools.partial(
    resample.resampling, resample.batched_multinomial
)


def predict_filter_update(
        model, proposal, update_state_fn,
        t0, graph, external,
        initial_histories, initial_states, initial_log_weights,
        initial_observations, initial_belief_states,
        horizon, num_samples, mode="PREDICT",
        swap_memory=False):
    _, initial_local_states = initial_states

    def cond(t, *unused_args):
        return tf.math.less(t, horizon)

    def body(t, histories, states, log_weights,
             observations, belief_states, arrays):
        locs_array, scales_array, log_probs_array = arrays
        locs, scales, log_probs = generative.predict(
            model, graph, external.current(t, t + 1),
            histories, states, observations, 1
        )
        squeeze = functools.partial(tf.squeeze, axis=[1])
        locs, scales, log_probs = \
            squeeze(locs), squeeze(scales), squeeze(log_probs)

        new_locs_array = locs_array.write(t, locs)
        new_scales_array = scales_array.write(t, scales)
        new_log_probs_array = log_probs_array.write(t, log_probs)
        new_arrays = (new_locs_array, new_scales_array, new_log_probs_array)

        new_observations = tf.math.reduce_mean(locs, axis=0)
        new_belief_states = proposal.refresh_beliefs(
            graph=graph, belief_states=belief_states,
            external=external.current(t), observations=new_observations
        )
        new_conditions = proposal.concat_conditions(
            external=external.current(t), observations=new_observations,
            beliefs=proposal.extract_summaries(new_belief_states),
            lookaheads=None
        )
        new_histories, new_states, new_log_weights = update_state_fn(
            mode=mode, t=(t0 + t),
            graph=graph, external=external.current(t),
            num_samples=num_samples,
            histories=histories, states=states, log_weights=log_weights,
            old_observations=observations,
            new_observations=new_observations,
            conditions=new_conditions,
            length=(t0 + t)
        )
        return (
            t + 1, new_histories, new_states, new_log_weights,
            new_observations, new_belief_states, new_arrays
        )

    initial_locs_array = tf.TensorArray(tf.float32, size=horizon)
    initial_scales_array = tf.TensorArray(tf.float32, size=horizon)
    initial_log_probs_array = tf.TensorArray(tf.float32, size=horizon)
    initial_arrays = (
        initial_locs_array, initial_scales_array, initial_log_probs_array
    )

    _, _, _, _, _, _, arrays = tf.while_loop(
        cond, body,
        [
            0, initial_histories, initial_states, initial_log_weights,
            initial_observations, initial_belief_states, initial_arrays
        ],
        swap_memory=swap_memory
    )
    locs_array, scales_array, log_probs_array = arrays
    locs, scales = locs_array.stack(), scales_array.stack()
    log_probs = log_probs_array.stack()

    assert locs.shape.ndims == 5
    assert scales.shape.ndims == 5
    assert log_probs.shape.ndims == 3

    static_batch_shape = initial_local_states.shape.as_list()[:-1]
    locs.set_shape([horizon, *static_batch_shape, None])
    scales.set_shape([horizon, *static_batch_shape, None])
    log_probs.set_shape([horizon, *static_batch_shape[:-1]])
    # (H, S, B, N, dx) -> (S, H, B, N, dx)
    locs = tf.transpose(locs, perm=[1, 0, 2, 3, 4])
    scales = tf.transpose(scales, perm=[1, 0, 2, 3, 4])
    # (H, S, B) -> (S, H, B)
    log_probs = tf.transpose(log_probs, perm=[1, 0, 2])

    return locs, scales, log_probs


def make_predict_fn(model, proposal, init_state_fn, update_state_fn,
                    parallel_iterations=10, swap_memory=False):

    def call(graph, external, observations, horizon, num_samples,
             every_step=False, resample_init=True, mode="PREDICT",
             auto_regressive_filtering=False,
             initial_belief_states=None,
             initial_latent_histories=None,
             initial_latent_states=None):
        '''
        Args:
          graph: A RuntimeGraph object.
          observations: A (T, B, N, dx) Tensor.
          horizon: Number of time steps to predict.
          num_samples: Number of Monte Carlo samples to use.
          every_step: A boolean.
          initial_belief_states: Optional. A (B, N, dh) Tensor.
          initial_latent_histories: Optional. A (S, B, N, dh) Tensor.
          initial_belief_states: Optional. A (S, B, N, dz) Tensor.

        Returns:
          predictions: A (T, S, H, B, N, dx) Tensor if every_step is True.
                       A (S, H, B, N, dx) Tensor else.
        '''
        assert observations.shape.ndims == 4
        num_observ_steps = tf.shape(observations)[0]

        truncated_external = external.truncate(
            num_observ_steps,
            total_length=(num_observ_steps + horizon)
        )
        beliefs, belief_states, latest_belief_states = \
            proposal.summarize_forward(
                graph=graph, external=truncated_external,
                sequence=observations,
                initial_states=initial_belief_states
            )
        lookaheads, _, _ = proposal.summarize_backward(
            graph=graph, external=truncated_external,
            sequence=observations,
            initial_states=latest_belief_states
        )
        lookaheads = util.left_shift_and_pad(
            lookaheads,
            proposal.extract_summaries(latest_belief_states)
        )
        conditions = proposal.concat_conditions(
            external=truncated_external,
            observations=observations,
            beliefs=beliefs, lookaheads=lookaheads
        )

        def _predict(t, histories, states, log_weights, arrays, widx):
            locs_array, scales_array, \
                log_probs_array, log_weights_array = arrays

            if resample_init:
                resampled_histories, resampled_states, _ = RESAMPLING(
                    histories=histories, particles=states,
                    log_weights=log_weights, num_samples=num_samples,
                    mask=graph.node_mask
                )
                histories, states = util.nested_set_shape_like(
                    (resampled_histories, resampled_states),
                    (histories, states)
                )
                log_weights = tf.zeros_like(log_weights)

            if auto_regressive_filtering:
                locs, scales, log_probs = predict_filter_update(
                    model=model, proposal=proposal,
                    update_state_fn=update_state_fn, t0=t,
                    graph=graph, external=external.current(t, t + horizon),
                    initial_histories=histories,
                    initial_states=states,
                    initial_log_weights=log_weights,
                    initial_observations=observations[t - 1],
                    initial_belief_states=belief_states[t - 1],
                    horizon=horizon, num_samples=num_samples,
                    mode=mode, swap_memory=swap_memory
                )
            else:
                locs, scales, log_probs = generative.predict(
                    model, graph, external.current(t, t + horizon),
                    histories, states, observations[t - 1], horizon
                )
            # (S, 1, B) + (S, H, B)
            log_probs = tf.math.add(
                tf.expand_dims(log_weights, axis=-2), log_probs
            )
            new_locs_array = locs_array.write(widx, locs)
            new_scales_array = scales_array.write(widx, scales)
            new_log_probs_array = log_probs_array.write(widx, log_probs)
            new_log_weights_array = log_weights_array.write(widx, log_weights)
            new_arrays = (
                new_locs_array, new_scales_array,
                new_log_probs_array, new_log_weights_array
            )
            return new_arrays

        def cond(t, *unused_args):
            return tf.less(t, num_observ_steps)

        def body(t, histories, states, log_weights, arrays):
            new_arrays = arrays if not every_step \
                else _predict(t, histories, states, log_weights, arrays, t - 1)

            new_histories, new_states, new_log_weights = update_state_fn(
                mode=mode, t=t, graph=graph, external=external.current(t),
                num_samples=num_samples,
                histories=histories, states=states, log_weights=log_weights,
                old_observations=observations[t - 1],
                new_observations=observations[t],
                conditions=conditions[t],
                length=num_observ_steps
            )
            return (
                t + 1, new_histories, new_states, new_log_weights, new_arrays
            )

        initial_histories, initial_states, initial_log_weights = \
            init_state_fn(
                mode=mode, graph=graph,
                external=external.current(0),
                observations=observations[0],
                conditions=conditions[0],
                num_samples=num_samples,
                length=num_observ_steps,
                initial_histories=initial_latent_histories,
                initial_states=initial_latent_states
            )
        array_size = 1 if not every_step else num_observ_steps
        initial_locs_array = tf.TensorArray(tf.float32, size=array_size)
        initial_scales_array = tf.TensorArray(tf.float32, size=array_size)
        initial_log_probs_array = tf.TensorArray(tf.float32, size=array_size)
        initial_log_weights_array = tf.TensorArray(tf.float32, size=array_size)
        initial_arrays = (
            initial_locs_array, initial_scales_array,
            initial_log_probs_array, initial_log_weights_array
        )
        t1 = tf.constant(1)

        end_t, final_histories, final_states, final_log_weights, \
            final_arrays = tf.while_loop(
                cond, body,
                [
                    t1, initial_histories, initial_states,
                    initial_log_weights, initial_arrays
                ],
                swap_memory=swap_memory
            )
        final_arrays = _predict(
            end_t, final_histories, final_states, final_log_weights,
            final_arrays, (0 if not every_step else (end_t - 1))
        )
        locs_array, scales_array, \
            log_probs_array, log_weights_array = final_arrays

        locs, scales = locs_array.stack(), scales_array.stack()
        log_probs = log_probs_array.stack()
        log_weights = log_weights_array.stack()

        squeeze = functools.partial(tf.squeeze, axis=[0])
        if not every_step:
            return (
                squeeze(locs), squeeze(scales),
                squeeze(log_probs), squeeze(log_weights)
            )
        return locs, scales, log_probs, log_weights

    return call
