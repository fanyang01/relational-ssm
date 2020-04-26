from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import generative
from auxiliary import compute_aux_scores
from proposal import collect_persistent_states
from predict import make_predict_fn

import functools

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Estimation(object):
    def __init__(self, vi_bound, iwae_bound,
                 distortion, divergence,
                 preview_divergence, aux_score):
        self.vi_bound = vi_bound
        self.iwae_bound = iwae_bound
        self.distortion = distortion
        self.divergence = divergence
        self.preview_divergence = preview_divergence
        self.aux_score = aux_score

    def install_summaries(self, collection, family):
        summ_scalar = functools.partial(
            tf.summary.scalar,
            collections=collection, family=family
        )
        summ_scalar("distortion", self.distortion)
        summ_scalar("divergence", self.divergence)
        summ_scalar("preview_divergence", self.preview_divergence)
        summ_scalar("aux_score", self.aux_score)
        summ_scalar("ELBO", self.vi_bound)
        summ_scalar("IWAE", self.iwae_bound)


def VI(gen_model, inf_model, aux_model=None,
       analytic_kl=False, num_preview_steps=0, pred_resample_init=True,
       parallel_iterations=128, swap_memory=True):

    dim_hidden = gen_model.dim_hidden

    def ELBO(graph, distortions, divergences, anneal_factor=1.0):
        '''
        Args:
          distortions: A (T, S, B) Tensor, -log p(x_t|z_t).
          divergences: A (T, S, B) Tensor, KL[q(z_t)||p(z_t|z_{<t})]
                                        or Monte Carlo estimations.

        Returns:
          elbo: A scalar Tensor.
        '''
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(distortions), tf.shape(divergences)),
            tf.assert_equal(distortions.shape.ndims, 3)
        ]):
            per_step_free_energy = tf.math.subtract(
                tf.math.negative(distortions),
                tf.multiply(anneal_factor, divergences)
            )
        # (T, S, B) -> (S, B) -> (B) -> scalar
        batched_elbo = tf.math.reduce_mean(
            tf.math.reduce_sum(per_step_free_energy, axis=0), axis=0
        )
        elbo = tf.reduce_mean(batched_elbo)
        # scaled_elbo = tf.math.multiply(
        #     tf.math.reduce_sum(
        #         tf.math.divide(batched_elbo, graph.num_nodes)
        #     ),
        #     graph.max_num_nodes
        # )
        # return scaled_elbo
        return elbo

    def IWAE(graph, distortions, divergences):
        '''
        Args:
          distortions: A (T, S, B) Tensor, -log p(x_t|z_t).
          divergences: A (T, S, B) Tensor, log q(z_t) - log p(z_t|z_{<t}).

        Returns:
          bound: A scalar Tensor.
        '''
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(distortions), tf.shape(divergences)),
            tf.assert_equal(distortions.shape.ndims, 3)
        ]):
            per_step_log_weights = tf.math.subtract(
                tf.math.negative(distortions), divergences
            )
        # (T, S, B) -> (B, S, T) -> (B, S)
        sis_log_weights = tf.math.reduce_sum(
            tf.transpose(per_step_log_weights, perm=[2, 1, 0]), axis=-1
        )
        # (B, S) -> (B)
        bound = tf.math.subtract(
            tf.math.reduce_logsumexp(sis_log_weights, axis=-1),
            tf.math.log(tf.to_float(tf.shape(sis_log_weights)[-1]))
        )
        return tf.math.reduce_mean(bound)

    def kl_divergence(graph, Q, P, samples):
        if analytic_kl:
            indep_kl = tfd.kl_divergence(Q, P)
        else:
            indep_kl = tf.math.subtract(
                Q.log_prob(samples), P.log_prob(samples)
            )

        # (S, B, N) * (B, N) -> (S, B, N)
        masked_indep_kl = tf.math.multiply(indep_kl, graph.center_mask)
        # (S, B, N) -> (S, B)
        return tf.math.multiply(
            tf.math.reduce_sum(masked_indep_kl, axis=-1),
            tf.math.divide(
                tf.to_float(graph.num_nodes),
                tf.math.reduce_sum(graph.center_mask, axis=-1)
            )
        )

    def masked_per_step_vi(
            mode, t, graph,
            new_histories, new_states, observations,
            prior, approx_posterior,
            distortions_array=None, divergences_array=None,
            preview_prior=None, preview_divergences_array=None,
            aux_fn=None, aux_scores_array=None,
            log_weights_acc=None):
        if new_states is None:
            new_states = approx_posterior.sample(1)[0]  # (S, B, N, dz)

        local_kl = kl_divergence(
            graph=graph, Q=approx_posterior, P=prior,
            samples=new_states
        )  # (S, B)
        if divergences_array is not None:
            divergences_array = divergences_array.write(t, local_kl)

        likelihood = gen_model.emit(
            histories=new_histories, states=new_states
        )
        indep_distortions = tf.math.negative(
            likelihood.log_prob(observations)
        )
        # (S, B, N) -> (S, B)
        local_distortion = graph.reduce_sum_nodal(indep_distortions)
        if distortions_array is not None:
            distortions_array = distortions_array.write(t, local_distortion)

        if log_weights_acc is not None:
            log_weights_acc = tf.math.add(log_weights_acc, tf.math.subtract(
                tf.math.negative(local_distortion), local_kl
            ))

        local_preview_kl = tf.zeros(tf.shape(new_states)[:-2])

        if num_preview_steps > 0 and preview_prior is not None:
            local_preview_kl = kl_divergence(
                graph=graph, Q=approx_posterior, P=preview_prior,
                samples=new_states
            )  # (S, B)
            local_preview_kl = tf.cond(
                tf.math.less(t, num_preview_steps),
                lambda: tf.zeros(tf.shape(new_states)[:-2]),
                lambda: local_preview_kl
            )
        if preview_divergences_array is not None:
            preview_divergences_array = preview_divergences_array.write(
                t, local_preview_kl
            )

        local_aux_score = tf.zeros(tf.shape(new_states)[:-2])

        if aux_fn is not None:
            local_aux_score = aux_fn(graph, new_histories, new_states)
        if aux_scores_array is not None:
            aux_scores_array = aux_scores_array.write(t, local_aux_score)

        # (S, B) -> (B) -> scalar
        def reduce_avg(x):
            return tf.math.reduce_mean(tf.math.divide(
                tf.math.reduce_mean(x, axis=0), tf.to_float(graph.num_nodes)
            ))

        print_op = tf.print(
            "mode iteration step kl distortion energy preview aux = ",
            mode, tf.train.get_global_step(), t,
            reduce_avg(local_kl),
            reduce_avg(local_distortion),
            reduce_avg(tf.math.negative(local_distortion + local_kl)),
            reduce_avg(local_preview_kl),
            reduce_avg(local_aux_score)
        )
        with tf.control_dependencies([print_op]):
            new_states = tf.identity(new_states)

        new_arrays = (
            distortions_array, divergences_array,
            preview_divergences_array, aux_scores_array
        )
        return new_states, log_weights_acc, new_arrays

    def make_aux_fn(beliefs, lookaheads):
        def call(graph, histories, states):
            return compute_aux_scores(
                aux_model, graph, histories, states,
                beliefs, lookaheads
            )
        return call

    def estimate(mode, graph, observations, num_samples,
                 anneal_factor=1.0,
                 inputs=None,
                 initial_belief_states=None,
                 initial_latent_states=None,
                 initial_latent_histories=None):
        '''
        Args:
          mode: A string tensor, 'TRAIN' or 'EVAL'.
          graph: A RuntimeGraph object.
          observations: A (T, B, N, dx) Tensor.
          num_samples: Number of Monte Carlo samples to use.
          anneal_factor: Optional. A scalar tensor.
          initial_belief_states: Optional. A (B, N, dh) Tensor.
          initial_belief_states: Optional. A (S, B, N, dz) Tensor.
          initial_latent_histories: Optional. A (S, B, N, dh) Tensor.

        Returns:
          elbo: A scalar tensor.
          distoration: A scalar tensor.
          divergence: A scalar tensor.
          last_belief_states: A (B, N, dh) Tensor.
          last_latent_states: A (S, B, N, dz) Tensor.
          last_latent_histories: A (S, B, N, dh) Tensor.
        '''
        num_steps = tf.shape(observations)[0]
        beliefs, final_belief_states = inf_model.summarize_forward(
            graph, observations, inputs=inputs,
            initial_states=initial_belief_states
        )
        lookaheads, _ = inf_model.summarize_backward(
            graph, observations, inputs=inputs
        )

        def cond(t, *unused_args):
            return tf.less(t, num_steps)

        def body(t, states, histories, previews, arrays):
            distortions, divergences, preview_divergences, aux_scores = arrays

            priors = gen_model.trans(
                graph=graph, histories=histories, states=states,
                inputs=current_inputs(inputs, t)
            )
            prior = priors.next_state_dist
            new_histories = priors.refreshed_histories
            approx_posterior = inf_model.propose(
                t=t, graph=graph, histories=histories, states=states,
                priors=priors,
                observations=observations[t],
                beliefs=beliefs[t], lookaheads=lookaheads[t],
                length=num_steps, inputs=current_inputs(inputs, t)
            )

            if num_preview_steps > 0:
                preview_concat_states = previews.read(t)
                preview_histories = preview_concat_states[..., :dim_hidden]
                preview_states = preview_concat_states[..., dim_hidden:]
                preview_priors = gen_model.trans(
                    graph=graph,
                    histories=preview_histories, states=preview_states,
                    inputs=current_inputs(inputs, t)  # TODO
                )
                preview_prior = preview_priors.next_state_dist
            else:
                preview_prior = None

            new_states, _, new_arrays = masked_per_step_vi(
                mode=mode, t=t, graph=graph,
                new_histories=new_histories, new_states=None,
                observations=observations[t],
                prior=prior, approx_posterior=approx_posterior,
                distortions_array=distortions, divergences_array=divergences,
                preview_prior=preview_prior,
                preview_divergences_array=preview_divergences,
                aux_fn=make_aux_fn(beliefs[t], lookaheads[t]),
                aux_scores_array=aux_scores
            )
            new_previews = previews.write(
                t + num_preview_steps, generative.preview(
                    gen_model, graph, new_states, new_histories,
                    num_preview_steps,
                    inputs=current_inputs(inputs, t, t + num_preview_steps)
                )
            )
            return (
                t + 1, new_states, new_histories,
                new_previews, new_arrays
            )

        assert (initial_latent_states is None) == \
            (initial_latent_histories is None)

        if initial_latent_states is None:
            init_approx_posterior = inf_model.init(
                graph=graph, observations=observations[0],
                beliefs=beliefs[0], lookaheads=lookaheads[0],
                length=num_steps
            )
            init_states = init_approx_posterior.sample(num_samples)
            init_histories = gen_model.init_histories_like(init_states)
            init_prior = gen_model.prior(tf.shape(init_states))
        else:
            priors = gen_model.trans(
                graph=graph,
                histories=initial_latent_histories,
                states=initial_latent_states,
                inputs=current_inputs(inputs, 0)
            )
            init_prior = priors.next_state_dist
            init_histories = priors.refreshed_histories
            init_approx_posterior = inf_model.propose(
                t=0, graph=graph,
                histories=initial_latent_histories,
                states=initial_latent_states,
                priors=priors,
                observations=observations[0],
                beliefs=beliefs[0], lookaheads=lookaheads[0],
                length=num_steps, inputs=current_inputs(inputs, 0)
            )
            init_states = None

        init_distortions = tf.TensorArray(tf.float32, size=num_steps)
        init_divergences = tf.TensorArray(tf.float32, size=num_steps)
        init_preview_divergences = tf.TensorArray(tf.float32, size=num_steps)
        init_aux_scores = tf.TensorArray(
            tf.float32, size=num_steps, clear_after_read=False)

        init_states, _, init_arrays = masked_per_step_vi(
            mode=mode, t=0, graph=graph,
            new_histories=init_histories, new_states=init_states,
            observations=observations[0],
            prior=init_prior, approx_posterior=init_approx_posterior,
            distortions_array=init_distortions,
            divergences_array=init_divergences,
            preview_prior=None,
            preview_divergences_array=init_preview_divergences,
            aux_fn=make_aux_fn(beliefs[0], lookaheads[0]),
            aux_scores_array=init_aux_scores
        )
        init_previews = generative.init_preview_array(
            num_steps, num_preview_steps,
            gen_model, graph, init_states, init_histories,
            inputs=current_inputs(inputs, 0, num_preview_steps)
        )
        t1 = tf.constant(1)

        _, final_states, final_histories, _, arrays = tf.while_loop(
            cond, body,
            [
                t1, init_states, init_histories,
                init_previews, init_arrays
            ],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory
        )
        distortions, divergences, preview_divergences, aux_scores = arrays
        distortions, divergences = distortions.stack(), divergences.stack()
        preview_divergences = preview_divergences.stack()
        aux_scores = aux_scores.stack()

        vi_bound = ELBO(
            graph=graph, distortions=distortions, divergences=divergences,
            anneal_factor=anneal_factor
        )
        iwae_bound = IWAE(
            graph=graph, distortions=distortions, divergences=divergences
        )

        # def scaled_reduce_sum(metrics):
        #     # (T, S, B) -> (T, B) -> (B) -> scalar
        #     batched_metrics = tf.math.reduce_sum(
        #         tf.math.reduce_mean(distortions, axis=-2), axis=0
        #     )
        #     return tf.math.multiply(
        #         tf.math.reduce_sum(graph.batch_avg(batched_metrics)),
        #         tf.to_float(graph.max_num_nodes)
        #     )
        def reduce_sum_avg(metrics):
            # (T, S, B) -> (T, B) -> (B) -> scalar
            return tf.math.reduce_mean(tf.math.reduce_sum(
                tf.math.reduce_mean(distortions, axis=-2), axis=0
            ), axis=0)

        avg_distortion = reduce_sum_avg(distortions)
        avg_divergence = reduce_sum_avg(divergences)
        avg_aux_score = reduce_sum_avg(aux_scores)
        avg_preview_divergence = reduce_sum_avg(preview_divergences)

        est = Estimation(
            vi_bound=vi_bound, iwae_bound=iwae_bound,
            distortion=avg_distortion, divergence=avg_divergence,
            preview_divergence=avg_preview_divergence,
            aux_score=avg_aux_score
        )
        states = collect_persistent_states(
            belief_states=final_belief_states,
            latent_states=final_states,
            latent_histories=final_histories
        )
        return est, states

    def init_state_fn(mode, graph, observations, beliefs, lookaheads,
                      num_samples, length, inputs=None):
        initial_approx_posterior = inf_model.init(
            graph=graph, observations=observations,
            beliefs=beliefs, lookaheads=lookaheads,
            length=length, inputs=inputs
        )
        initial_prior = gen_model.prior()
        initial_states = initial_approx_posterior.sample(num_samples)
        initial_histories = gen_model.init_histories_like(initial_states)
        initial_log_weights = tf.zeros(tf.shape(initial_states)[:-2])

        initial_states, initial_log_weights, _ = masked_per_step_vi(
            mode=mode, t=tf.constant(0), graph=graph,
            new_histories=initial_histories, new_states=initial_states,
            observations=observations,
            prior=initial_prior, approx_posterior=initial_approx_posterior,
            log_weights_acc=initial_log_weights
        )
        return initial_states, initial_histories, initial_log_weights

    def update_state_fn(mode, t, graph, num_samples,
                        states, histories, log_weights,
                        observations, beliefs, lookaheads,
                        length, inputs=None):
        priors = gen_model.trans(
            graph=graph, histories=histories, states=states,
            inputs=inputs
        )
        prior = priors.next_state_dist
        new_histories = priors.refreshed_histories
        approx_posterior = inf_model.propose(
            t=t, graph=graph, histories=histories, states=states,
            observations=observations, inputs=inputs,
            beliefs=beliefs, lookaheads=lookaheads,
            priors=priors, length=length
        )
        new_states, new_log_weights, _ = masked_per_step_vi(
            mode=mode, t=t, graph=graph,
            new_histories=new_histories, new_states=None,
            observations=observations,
            prior=prior, approx_posterior=approx_posterior,
            log_weights_acc=log_weights
        )
        return new_states, new_histories, new_log_weights

    predict = make_predict_fn(
        model=gen_model, proposal=inf_model,
        init_state_fn=init_state_fn, update_state_fn=update_state_fn,
        parallel_iterations=parallel_iterations, swap_memory=swap_memory
    )

    return estimate, predict
