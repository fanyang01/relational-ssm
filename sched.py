from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import util

import tensorflow as tf


class InterleavingScheduler(object):
    def __init__(self, prefix_length=1, rate=0.5,
                 randomly=True, refresh_last_step=False):
        assert prefix_length > 0
        self.prefix_length = prefix_length
        self.refresh_last_step = refresh_last_step
        self.randomly = randomly
        self.rate = rate
        self.period = tf.cast(tf.math.ceil(1.0 / rate), tf.int32)

    def sched(self, context, batch_size):
        t, length = context.t, context.length

        if self.randomly:
            use_proposal = tf.math.less(
                tf.random.uniform(shape=[batch_size], dtype=tf.float32),
                self.rate
            )
        else:
            use_proposal = tf.math.equal(
                tf.floormod(t, self.period),
                tf.zeros([batch_size], dtype=tf.int32)
            )

        use_proposal = tf.math.logical_or(
            use_proposal,
            tf.math.logical_and(
                tf.math.equal(t, length - 1),
                self.refresh_last_step
            )
        )
        use_proposal = tf.math.logical_or(
            use_proposal,
            tf.math.less(t, self.prefix_length)
        )
        context.use_proposal = use_proposal
        return util.float(use_proposal)
