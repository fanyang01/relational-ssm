from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def reduce_avg(predictions):
    # ([T, ]S, H, B, N, dx) -> ([T, ]H, B, N, dx)
    return tf.math.reduce_mean(predictions, axis=-5)


def reduce_wavg(predictions, log_weights):
    assert len(log_weights.shape.as_list()) == 2
    normalized_weights = tf.math.softmax(log_weights, axis=0)
    # ([T, ]S, B) -> ([T, ]S, 1, B) -> ([T, ]S, 1, B, 1, 1)
    normalized_weights = tf.expand_dims(normalized_weights, axis=-2)
    expand = functools.partial(tf.expand_dims, axis=-1)
    normalized_weights = expand(expand(normalized_weights))

    # ([T, ]S, H, B, N, dx) * ([T, ]S, 1, B, 1, 1) -> ([T, ]H, B, N, dx)
    return tf.math.reduce_sum(
        tf.math.multiply(predictions, normalized_weights), axis=-5
    )


def reduce_median(predictions):
    # ([T, ]S, H, B, N, dx) -> ([T, ]H, B, N, dx)
    return tfp.stats.percentile(predictions, q=50.0, axis=-5)


def reduce_max_prob(predictions, log_probs):
    assert log_probs.shape.ndims >= 3
    # ([T, ]S, H, B) -> ([T, ]H, B) -> ([T, ]H, B, 1)
    indices = tf.expand_dims(
        tf.math.argmax(log_probs, axis=-3, output_type=tf.int32),
        axis=-1
    )
    # ([T, ]S, H, B, N, dx) -> ([T, ]H, B, S, N, dx)
    perm = tf.range(predictions.shape.ndims)
    perm = tf.stack([
        *tf.unstack(perm[:-5]),
        perm[-4], perm[-3], perm[-5], perm[-2], perm[-1]
    ])
    predictions = tf.transpose(predictions, perm)
    # ([T, ]H, B, S, N, dx) --gather[([T, ]H, B, 1)]--> ([T, ]H, B, 1, N, dx)
    max_prob_predictions = tf.batch_gather(predictions, indices)
    return tf.squeeze(max_prob_predictions, axis=-3)


def reduce_mode(predict_locs, predict_scales):
    # ([T, ]S, H, B, N, dx) -> ([T, ]H, B, S, N, dx)
    perm = tf.range(predict_locs.shape.ndims)
    perm = tf.stack([
        *tf.unstack(perm[:-5]),
        perm[-4], perm[-3], perm[-5], perm[-2], perm[-1]
    ])
    # ([T, ]H, B, S, N, dx) -> ([T, ]H, B, S, {N, dx})
    predict_locs = tf.transpose(predict_locs, perm)
    predict_scales = tf.transpose(predict_scales, perm)
    dist = tfd.MultivariateNormalDiag(
        loc=predict_locs, scale_diag=predict_scales)
    indep_dist = tfd.Independent(dist, reinterpreted_batch_ndims=1)
    # ([T, ]H, B, S, {N, dx}) -> ([T, ]H, B, {N, dx})
    num_samples = tf.shape(predict_locs)[-3]
    mixture_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=tf.zeros([num_samples])),
        components_distribution=indep_dist
    )
    # ([T, ]H, B, S, N, dx) -> (S, [T, ]H, B, N, dx)
    perm = tf.range(predict_locs.shape.ndims)
    perm = tf.stack([
        perm[-3],
        *tf.unstack(perm[:-5]),
        perm[-5], perm[-4], perm[-2], perm[-1]
    ])
    tmp_predict_locs = tf.transpose(predict_locs, perm)
    # (S, [T, ]H, B) -> ([T, ]H, B) ->  ([T, ]H, B, 1)
    log_probs = mixture_dist.log_prob(tmp_predict_locs)
    indices = tf.math.argmax(log_probs, axis=0, output_type=tf.int32)
    indices = tf.expand_dims(indices, axis=-1)
    # ([T, ]H, B, S, N, dx) --gather[([T, ]H, B, 1)]--> ([T, ]H, B, 1, N, dx)
    modes = tf.batch_gather(predict_locs, indices)
    return tf.squeeze(modes, axis=-3)


def _check_shape(predictions, labels):
    with tf.control_dependencies([
        tf.assert_equal(tf.shape(predictions), tf.shape(labels))
    ]):
        predictions = tf.identity(predictions)
        labels = tf.identity(labels)
    return predictions, labels


def _mean_error(difference):
    # ([T, ]H, B, N, dx) -> ([T, ]H, dx)
    err = tf.math.reduce_mean(difference, axis=[-3, -2])
    if len(difference.shape.as_list()) == 5:
        # (T, H, dx) -> (H, dx)
        err = tf.math.reduce_mean(err, axis=0)
    if err.shape.as_list()[-1] == 1:
        return tf.squeeze(err, axis=[-1])
    return err


def MSE(predictions, labels):
    '''
    Args:
      predictions: A ([T, ]H, B, N, dx) Tensor.
      labels: A ([T, ]H, B, N, dx) Tensor.

    Returns:
      err: A (H, dx) or a (H) (if dx == 1) Tensor.
    '''
    predictions, labels = _check_shape(predictions, labels)
    diff = tf.math.squared_difference(predictions, labels)
    return _mean_error(diff)


def MAE(predictions, labels):
    predictions, labels = _check_shape(predictions, labels)
    diff = tf.math.abs(tf.math.subtract(predictions, labels))
    return _mean_error(diff)


def MAPE(predictions, labels):
    predictions, labels = _check_shape(predictions, labels)
    diff = tf.math.abs(
        tf.div_no_nan(
            tf.math.subtract(predictions, labels),
            labels
        )
    )
    return _mean_error(diff)


def PICP(predictions, labels):
    assert (predictions.shape.ndims - 1) == labels.shape.ndims
    # ([T, ]S, H, B, N, dx) -> ([T, ]H, B, N, dx)
    lower = tfp.stats.percentile(predictions, q=5.0, axis=-5)
    upper = tfp.stats.percentile(predictions, q=95.0, axis=-5)
    in_interval = tf.math.logical_and(
        tf.math.greater_equal(labels, lower),
        tf.math.less_equal(labels, upper)
    )
    return _mean_error(tf.cast(in_interval, tf.float32))
