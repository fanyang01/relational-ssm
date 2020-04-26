from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util import mlp_two_layers

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.ops.linalg import linear_operator_util as linalg_util


class MatvecQR(tfp.bijectors.bijector.Bijector):
    def __init__(self, Q, R, validate_args=False, name=None):
        '''
        Creates the MatvecQR bijector.

        Args:
          validate_args: Python `bool` indicating whether arguments should be
            checked for correctness.
            Default value: `False`.
          name: Python `str` name given to ops managed by this object.
            Default value: `None` (i.e., "MatvecQR").
        '''
        assert Q.shape.ndims == 2 and R.shape.ndims == 2
        Q_inv = tf.linalg.transpose(Q)
        self._Q = Q
        self._R = R
        self._Q_operator = tf.linalg.LinearOperatorFullMatrix(Q)
        self._Q_inv_operator = tf.linalg.LinearOperatorFullMatrix(Q_inv)

        super(MatvecQR, self).__init__(
            is_constant_jacobian=True,
            forward_min_event_ndims=1,
            validate_args=validate_args,
            name=name
        )

    @property
    def Q(self):
        return self._Q

    @property
    def R(self):
        return self._R

    def _forward(self, x):
        return self._Q_operator.matvec(
            linalg_util.matmul_with_broadcast(
                self._R, x[..., tf.newaxis]
            )[..., 0]
        )

    def _inverse(self, y):
        return linalg_util.matrix_triangular_solve_with_broadcast(
            matrix=self._R,
            rhs=self._Q_inv_operator.matvec(y)[..., tf.newaxis],
            lower=False, adjoint=False
        )[..., 0]

    def _forward_log_det_jacobian(self, unused_x):
        return tf.math.reduce_sum(
            tf.math.log(tf.math.abs(tf.linalg.diag_part(self._R))),
            axis=-1
        )


def real_nvp_default_fn(dim_in, dim_out, activation=tf.math.tanh, name=None):
    with tf.variable_scope(name or "real_nvp_default_fn"):
        mlp = mlp_two_layers(
            dim_in=dim_in,
            dim_hid=(4 * dim_in),
            dim_out=(2 * dim_out),
            act_out=activation,
            weight_init='small'
        )

    def _fn(z, units, **condition_kwargs):
        if condition_kwargs:
            raise NotImplementedError(
                "Conditioning not implemented in the default fn.")

        if z.shape.ndims == 1:
            z = z[tf.newaxis, ...]
            reshape_output = lambda x: x[0]
        else:
            reshape_output = lambda x: x

        y = mlp(z)
        shift, log_scale = tf.split(y, 2, axis=-1)
        return reshape_output(shift), reshape_output(log_scale)

    return _fn
