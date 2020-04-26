from __future__ import absolute_import
from __future__ import print_function

import util

import functools
import tensorflow as tf


def tf_pad_axis_to(a, axis, size, const=None):
    shape = tf.shape(a)
    with tf.control_dependencies([tf.assert_less_equal(shape[axis], size)]):
        shape = tf.stack([
            *tf.unstack(shape[:axis]),
            size - shape[axis],
            *(tf.unstack(shape[(axis + 1):]) if axis != -1 else [])
        ])
    if const is not None:
        return tf.concat([a, tf.fill(shape, const)], axis=axis)
    return tf.concat([a, tf.zeros(shape, dtype=a.dtype)], axis=axis)


def tf_lexsort(indices_2d):
    '''
    Args:
      indices_2d: A (E, 2) Tensor.

    Returns:
      lex_sorted: A (E, 2) Tensor.
      indices: A (E) Tensor.
    '''
    ids = tf.range(tf.shape(indices_2d)[0], dtype=tf.int32)
    flatten = tf.reshape(indices_2d, [-1])
    maximum = tf.cast(flatten[tf.math.argmax(flatten)], tf.int64)
    sp = tf.SparseTensor(
        indices=tf.cast(indices_2d, tf.int64), values=ids,
        dense_shape=tf.stack([maximum, maximum])
    )
    sp = tf.sparse.reorder(sp)
    lex_sorted, indices = tf.cast(sp.indices, indices_2d.dtype), sp.values
    with tf.control_dependencies([
        tf.assert_equal(tf.shape(indices_2d), tf.shape(lex_sorted)),
        tf.assert_equal(tf.gather(indices_2d, indices), lex_sorted)
    ]):
        return tf.identity(lex_sorted), tf.identity(indices)


class BaseRuntimeGraph(object):
    def __init__(self, edges, center_mask, node_mask, edge_mask,
                 dense=False, node_attrs=None, edge_attrs=None,
                 reversed=None):
        batch_size = tf.shape(edges)[0]
        num_nodes = tf.math.reduce_sum(node_mask, axis=-1)  # (B, N) -> (B)
        max_num_nodes = num_nodes[tf.math.argmax(num_nodes)]
        num_edges = tf.math.reduce_sum(edge_mask, axis=-1)  # (B, E) -> (B)
        max_num_edges = num_edges[tf.math.argmax(num_edges)]

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(node_mask), tf.shape(center_mask)),
            tf.assert_equal(tf.shape(edges)[:-1], tf.shape(edge_mask)),
            tf.assert_equal(tf.shape(node_mask)[-1], max_num_nodes),
            tf.assert_equal(tf.shape(edge_mask)[-1], max_num_edges),
            tf.assert_equal(tf.size(num_nodes), batch_size)
        ]):
            sparse_edges, batch_edge_indices = self._gen_sparse_edges(
                batched_edges=edges, batched_edge_mask=edge_mask
            )
            adjs, edges = self._gen_sparse_adj_matrix(
                sparse_edges=sparse_edges, batched_edges=edges,
                batched_edge_mask=edge_mask, max_num_nodes=max_num_nodes
            )

        # (B, N, N) -> (B, N)
        indegree = tf.sparse.reduce_sum(adjs, axis=-2)
        outdegree = tf.sparse.reduce_sum(adjs, axis=-1)
        # (B, N) -- batch_gather([B, E]) --> (B, E)
        tail_indegree = tf.batch_gather(indegree, indices=edges[..., 1])
        head_outdegree = tf.batch_gather(outdegree, indices=edges[..., 0])

        if node_attrs is not None and node_attrs.shape.ndims == 2:
            with tf.control_dependencies([
                tf.assert_equal(tf.shape(node_attrs)[0], max_num_nodes)
            ]):
                node_attrs = tf.tile(
                    tf.expand_dims(node_attrs, axis=0),
                    tf.stack([batch_size, 1, 1])
                )

        self._batch_size = batch_size
        self._adjs = adjs
        self._edges = tf.cast(edges, tf.int32)
        self._edges_int64 = tf.cast(edges, tf.int64)
        self._senders = self._edges[..., 0]
        self._receivers = self._edges[..., 1]
        self._sparse_edges = tf.cast(sparse_edges, tf.int64)
        self._batch_edge_indices = tf.cast(batch_edge_indices, tf.int32)
        self._node_attrs = node_attrs
        self._edge_attrs = edge_attrs
        self._num_nodes = num_nodes
        self._num_edges = num_edges
        self._max_num_nodes = max_num_nodes
        self._max_num_edges = max_num_edges
        self._total_num_nodes = tf.math.reduce_sum(num_nodes, axis=-1)
        self._total_num_edges = tf.math.reduce_sum(num_edges, axis=-1)
        self._indegree = indegree
        self._outdegree = outdegree
        self._tail_indegree = util.float(tail_indegree)
        self._head_outdegree = util.float(head_outdegree)
        self._center_mask = util.float(center_mask)
        self._node_mask = util.float(node_mask)
        self._edge_mask = util.float(edge_mask)
        self._reversed = reversed

        self._dense_adjs = self._dense_edge_attrs = None
        if dense:
            self._dense_adjs = tf.sparse.to_dense(adjs)
            self._dense_edge_attrs = self._edge_attrs_to_dense(
                edges, edge_attrs, max_num_nodes
            )
            # print_op = tf.print("dense_adjs:", self._dense_adjs)
            # with tf.control_dependencies([print_op]):
            #     self._dense_adjs = tf.identity(self._dense_adjs)

    def _gen_sparse_edges(self, batched_edges, batched_edge_mask):
        batch_size = tf.shape(batched_edge_mask)[0]
        max_num_edges = tf.shape(batched_edge_mask)[1]
        num_edges = tf.math.reduce_sum(batched_edge_mask, axis=-1)  # (B)

        # (E) -> (1, E) -> (B, E)
        edge_ids = tf.tile(
            tf.expand_dims(tf.range(max_num_edges), axis=0),
            [batch_size, 1]
        )
        batch_edge_indices = tf.where(tf.less(
            edge_ids, tf.expand_dims(num_edges, axis=-1)
        ))  # (\sum_b Eb, 2)

        # (B) -> (B, 1) -> (B, E) -> (B, E, 1)
        batch_ids = tf.expand_dims(
            tf.math.add(
                tf.expand_dims(tf.range(batch_size), axis=-1),
                tf.zeros_like(batched_edge_mask)
            ), axis=-1
        )
        # -> (B, E, 3) -> (\sum_b Eb, 3)
        sparse_edges = tf.gather_nd(
            tf.concat([batch_ids, batched_edges], axis=-1),
            indices=batch_edge_indices
        )
        return sparse_edges, batch_edge_indices

    def _gen_sparse_adj_matrix(self, sparse_edges, batched_edges,
                               batched_edge_mask, max_num_nodes):
        '''
        Args:
          sparse_edges: A (TE, 3) Tensor.
          batched_edges: A (B, E, 2) Tensor.
          batched_edge_mask: A (B, E) Tensor.
          max_num_nodes: A scalar Tensor.
        '''
        batched_edges = tf.cast(batched_edges, tf.int64)
        batch_size = tf.cast(tf.shape(batched_edges)[0], tf.int64)
        batched_num_edges = tf.math.reduce_sum(batched_edge_mask, axis=-1)
        max_num_nodes = tf.cast(max_num_nodes, tf.int64)
        max_num_edges = batched_num_edges[tf.math.argmax(batched_num_edges)]

        # (B) -> (B, 1, 1) + (B, E, 1) -> (B, E, 1)
        expand = functools.partial(tf.expand_dims, axis=-1)
        batch_ids = tf.math.add(
            expand(expand(tf.range(batch_size, dtype=tf.int64))),
            expand(tf.zeros(tf.shape(batched_edges)[:-1], dtype=tf.int64))
        )

        # (B, E, 1 + 2) -> (B, E, 3) -> (B*E, 3) -> (TE, 3): [[bid, sid, rid]]
        sparse_indices = tf.concat([batch_ids, batched_edges], axis=-1)
        sparse_indices = tf.reshape(sparse_indices, shape=tf.stack([
            tf.math.reduce_prod(tf.shape(batched_edges)[:-1]), 3
        ]))

        # (B, N, N)
        sp_adj_shape = tf.stack([batch_size, max_num_nodes, max_num_nodes])

        # Eliminate duplicated indices.
        # https://stackoverflow.com/questions/38233821/merge-duplicate-indices-in-a-sparse-tensor
        linearized_indices = tf.linalg.matmul(
            tf.cast(sparse_indices, tf.int32),
            tf.expand_dims(tf.math.cumprod(
                tf.cast(sp_adj_shape, tf.int32),
                reverse=True, exclusive=True
            ), axis=-1)
        )
        uid, _ = tf.unique(tf.squeeze(linearized_indices))
        uid = tf.cast(uid, tf.int64)
        bid = tf.math.floordiv(uid, max_num_nodes * max_num_nodes)
        batch_internal_id = tf.floormod(uid, max_num_nodes * max_num_nodes)
        sid = tf.math.floordiv(batch_internal_id, max_num_nodes)
        rid = tf.floormod(batch_internal_id, max_num_nodes)
        sparse_indices = tf.stack([bid, sid, rid], axis=-1)

        ones = tf.ones([tf.shape(sparse_indices)[0]], dtype=tf.int8)
        adj = tf.sparse.SparseTensor(
            indices=sparse_indices, values=ones, dense_shape=sp_adj_shape
        )
        # Sort the edges in canonical ordering
        adj = tf.sparse.reorder(adj)
        sparse_indices = adj.indices

        # Compare the indices generated in two different ways
        with tf.control_dependencies([
            tf.assert_equal(tf.cast(sparse_edges, tf.int64), sparse_indices)
        ]):
            sparse_indices = tf.identity(sparse_indices)

        # Check the correctness of input data CAREFULLY :)
        def fn(params):
            bid, start, num_edges = params
            indices = tf.range(start=start, limit=start + num_edges)
            edges = tf.gather(sparse_indices, indices)
            with tf.control_dependencies([tf.assert_equal(
                edges[:, 0], tf.ones([num_edges], dtype=tf.int64) * bid
            )]):
                return tf.concat([
                    edges[:, 1:],
                    tf.zeros([max_num_edges - num_edges, 2], dtype=tf.int64)
                ], axis=0)

        cum_num_edges = tf.math.cumsum(batched_num_edges, exclusive=True)
        batch_ids = tf.range(batch_size, dtype=tf.int64)
        padded_edges = tf.map_fn(
            fn, (batch_ids, cum_num_edges, batched_num_edges),
            dtype=tf.int64
        )
        with tf.control_dependencies([
            tf.assert_equal(
                tf.shape(sparse_indices)[0],
                tf.math.reduce_sum(batched_num_edges)
            ),
            tf.assert_equal(batched_edges, padded_edges)
        ]):
            padded_edges = tf.identity(tf.cast(padded_edges, tf.int32))
        return adj, padded_edges

    def _edge_attrs_to_dense(self, batched_edges, batched_edge_attrs,
                             max_num_nodes):
        '''
        Args:
          batched_edges: A (B, E, 2) Tensor.
          batched_edge_attrs: A (B, E, de) Tensor.

        Returns:
          dense_edge_attrs: A (B, N, N, de) Tensor.
        '''
        if batched_edge_attrs is None:
            return None
        dim_edge_attr = batched_edge_attrs.shape.as_list()[-1]

        def sparse_to_dense(params):
            edges, edge_attrs = params
            shape = [max_num_nodes, max_num_nodes, dim_edge_attr]
            return tf.manip.scatter_nd(
                indices=edges, updates=edge_attrs, shape=shape
            )

        dense_edge_attrs = tf.map_fn(
            sparse_to_dense, (batched_edges, batched_edge_attrs),
            dtype=tf.float32
        )
        return dense_edge_attrs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def total_num_nodes(self):
        return self._total_num_nodes

    @property
    def total_num_edges(self):
        return self._total_num_edges

    @property
    def max_num_nodes(self):
        return self._max_num_nodes

    @property
    def max_num_edges(self):
        return self._max_num_edges

    @property
    def edges(self):
        return self._edges

    @property
    def edges_int64(self):
        return self._edges_int64

    @property
    def senders(self):
        return self._senders

    @property
    def receivers(self):
        return self._receivers

    @property
    def sparse_edges(self):
        ''' A (TE, 3) Tensor. '''
        return self._sparse_edges

    @property
    def batch_edge_indices(self):
        ''' A (TE, 2) Tensor. '''
        return self._batch_edge_indices

    @property
    def node_attrs(self):
        return self._node_attrs

    @property
    def edge_attrs(self):
        return self._edge_attrs

    @property
    def adjacency(self):
        return self._adjs

    @property
    def dense_adjacency(self):
        return self._dense_adjs

    @property
    def dense_edge_attrs(self):
        return self._dense_edge_attrs

    @property
    def indegree(self):
        return self._indegree

    @property
    def outdegree(self):
        return self._outdegree

    @property
    def tail_indegree(self):
        return self._tail_indegree

    @property
    def head_outdegree(self):
        return self._head_outdegree

    @property
    def center_mask(self):
        return self._center_mask

    @property
    def node_mask(self):
        return self._node_mask

    @property
    def edge_mask(self):
        return self._edge_mask

    def dense_edge_weights_to_sparse(self, dense_edge_weights):
        '''
        Args:
          dense_edge_weights: A (..., B, E) Tensor.

        Returns:
          sparse_edge_weights: A (..., B, N, N) SparseTensor.
        '''
        with tf.control_dependencies([tf.assert_equal(
            tf.shape(dense_edge_weights)[-2:], tf.shape(self.edge_mask)
        )]):
            prefix_shape = tf.shape(dense_edge_weights)[:-2]

        prefix_size = tf.math.reduce_prod(prefix_shape)
        # (..., B, E) -> (P, B, E)
        flat_edge_weights = tf.reshape(dense_edge_weights, shape=tf.stack([
            prefix_size, self.batch_size, self.max_num_edges
        ]))
        # (TE, 2/3) -> (P, TE, 2/3)
        tiled_indices = tf.tile(
            tf.expand_dims(self.batch_edge_indices, axis=0),
            tf.stack([prefix_size, 1, 1])
        )
        tiled_sparse_positions = tf.tile(
            tf.expand_dims(self.sparse_edges, axis=0),
            tf.stack([prefix_size, 1, 1])
        )

        # (P, TE, 1 + 2/3) -> (P*TE, 3/4)
        prefix_ids = tf.tile(
            tf.expand_dims(tf.range(prefix_size), axis=-1),
            tf.stack([1, self.total_num_edges])
        )
        indices_3d = tf.concat([
            tf.expand_dims(prefix_ids, axis=-1), tiled_indices
        ], axis=-1)
        sparse_positions_4d = tf.concat([
            tf.expand_dims(tf.cast(prefix_ids, tf.int64), axis=-1),
            tiled_sparse_positions
        ], axis=-1)
        flat_indices_3d = tf.reshape(indices_3d, [-1, 3])
        flat_sparse_indices_4d = tf.reshape(sparse_positions_4d, [-1, 4])

        dense_shape_suffix = [
            self.batch_size, self.max_num_nodes, self.max_num_nodes
        ]
        sparse_tensor = tf.SparseTensor(
            indices=flat_sparse_indices_4d,
            values=tf.gather_nd(flat_edge_weights, flat_indices_3d),
            dense_shape=tf.cast(
                tf.stack([prefix_size, *dense_shape_suffix]), tf.int64
            )
        )
        return tf.sparse.reshape(sparse_tensor, shape=tf.stack([
            *tf.unstack(prefix_shape), *dense_shape_suffix
        ]))

    def sparse_edge_weights_to_dense(self, sparse_edge_weights):
        '''
        Args:
          sparse_edge_weights: A (..., B, N, N) SparseTensor.

        Returns:
          dense_edge_weights: A (..., B, E) Tensor.
        '''
        sparse_dense_shape = sparse_edge_weights.dense_shape
        with tf.control_dependencies([tf.assert_equal(
            tf.cast(sparse_dense_shape[-3:], tf.int32),
            tf.stack([
                self.batch_size, self.max_num_nodes, self.max_num_nodes
            ])
        )]):
            prefix_shape = tf.cast(sparse_dense_shape[:-3], tf.int32)
            prefix_size = tf.math.reduce_prod(prefix_shape)

        # (TE, 2) -> (P, TE, 2)
        tiled_indices = tf.tile(
            tf.expand_dims(self.batch_edge_indices, axis=0),
            tf.stack([prefix_size, 1, 1])
        )

        # (P, TE, 1 + 2) -> (P*TE, 3)
        prefix_ids = tf.tile(
            tf.expand_dims(tf.range(prefix_size), axis=-1),
            tf.stack([1, self.total_num_edges])
        )
        indices_3d = tf.concat([
            tf.expand_dims(prefix_ids, axis=-1), tiled_indices
        ], axis=-1)
        flat_indices_3d = tf.reshape(indices_3d, [-1, 3])

        flat_shape = tf.stack([
            prefix_size, self.batch_size, self.max_num_edges
        ])
        dense_shape = tf.stack([
            *tf.unstack(prefix_shape), self.batch_size, self.max_num_edges
        ])
        flat_edge_weights = tf.scatter_nd(
            indices=flat_indices_3d,
            updates=sparse_edge_weights.values,
            shape=flat_shape
        )
        return tf.reshape(flat_edge_weights, dense_shape)

    def gen_adj_mask_like(self, weights, reverse_mask=None, transpose=False):
        '''
        Args:
          weights: A (..., B, N, N) Tensor.
          transpose: If TRUE, (..., B, sid, rid) -> (..., B, rid, sid).
          reverse_mask: Optional. A ([..., ]B, N, N) Tensor, True => DISABLE.

        Returns:
          mask: A (..., B, N, N) Tensor, True => EDGE.
        '''
        assert self.dense_adjacency is not None
        mask = tf.math.logical_and(
            tf.fill(tf.shape(weights), tf.constant(True)),
            tf.math.not_equal(
                self.dense_adjacency,
                tf.constant(0, dtype=tf.int8)
            )
        )
        if reverse_mask is not None:
            mask = tf.math.logical_and(mask, tf.math.logical_not(reverse_mask))
        return mask if transpose else tf.linalg.transpose(mask)

    def gen_dense_self_loop_mask(self, batch_shape):
        with tf.control_dependencies([
            tf.assert_equal(batch_shape[-1], self.max_num_nodes)
        ]):
            batch_adj_shape = tf.stack([
                *tf.unstack(batch_shape), self.max_num_nodes
            ])
        return tf.math.logical_and(
            tf.fill(batch_adj_shape, tf.constant(True)),
            tf.math.equal(
                tf.eye(self.max_num_nodes, dtype=tf.int8),
                tf.constant(1, dtype=tf.int8)
            )
        )

    def gen_sparse_self_loop_mask(self, batch_shape):
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(self.node_mask), batch_shape[-2:])
        ]):
            batch_edge_shape = tf.stack([
                *tf.unstack(batch_shape[:-1]), self.max_num_edges
            ])
        mask = tf.math.equal(self.senders, self.receivers)
        return tf.math.logical_and(
            mask, tf.fill(batch_edge_shape, tf.constant(True)),
        )

    def gen_dense_recv_only_mask(self, node_ids):
        '''
        Args:
          node_ids: A ([..., ]B, K) Tensor.

        Returns:
          adj_mask: A ([..., ]B, N, N) Tensor, True => MASKED.
          node_mask: A ([..., ]B, N) Tensor, True => MASKED.
        '''
        prefix_shape = tf.shape(node_ids)[:-2]
        prefix_size = tf.math.reduce_prod(prefix_shape)
        K = node_ids.shape.as_list()[-1]
        assert K is not None
        expand = functools.partial(tf.expand_dims, axis=-1)

        # (..., B, K) -> (P, B, K)
        flat_node_ids = tf.reshape(
            node_ids, shape=tf.stack([prefix_size, self.batch_size, K])
        )
        # (P) -> (P, 1, 1) -> (P, B, K)
        prefix_ids = tf.math.add(
            tf.zeros_like(flat_node_ids), expand(expand(tf.range(prefix_size)))
        )
        # (B) -> (B, 1) -> (P, B, K)
        batch_ids = tf.math.add(
            tf.zeros_like(flat_node_ids), expand(tf.range(self.batch_size))
        )
        # -> (P, B, K, 1 + 1 + 1)
        mask_indices = tf.concat([
            expand(prefix_ids), expand(batch_ids), expand(flat_node_ids)
        ], axis=-1)

        flat_mask = tf.scatter_nd(
            indices=mask_indices,
            updates=tf.ones_like(flat_node_ids, dtype=tf.int8),
            shape=tf.stack([prefix_size, self.batch_size, self.max_num_nodes])
        )
        mask = tf.reshape(flat_mask, shape=tf.stack([
            *tf.unstack(prefix_shape), self.batch_size, self.max_num_nodes
        ]))
        node_mask = tf.math.not_equal(mask, tf.constant(0, dtype=tf.int8))

        if self.dense_adjacency is None:
            return None, node_mask
        adj_mask = tf.math.logical_and(
            tf.fill(tf.shape(self.dense_adjacency), tf.constant(True)),
            tf.expand_dims(node_mask, axis=-1)
        )
        return adj_mask, node_mask

    def gen_sparse_recv_only_mask(self, node_ids):
        '''
        Args:
          node_ids: A ([..., ]B, K) Tensor.

        Returns:
          recv_only_mask: A ([..., ]B, E) Tensor, True => MASKED.
        '''
        prefix_shape = tf.shape(node_ids)[:-2]
        prefix_size = tf.math.reduce_prod(prefix_shape)
        K = node_ids.shape.as_list()[-1]
        assert K is not None
        expand = functools.partial(tf.expand_dims, axis=-1)

        # (..., B, K) -> (P, B, K)
        flat_node_ids = tf.reshape(
            node_ids, shape=tf.stack([prefix_size, self.batch_size, K])
        )
        # (B, E) -> (P, B, E)
        broadcast_sender_ids = tf.tile(
            tf.expand_dims(self.senders, axis=0), [prefix_size, 1, 1]
        )

        # (P, B, K)
        left_idx = tf.searchsorted(
            broadcast_sender_ids, flat_node_ids, side="left")
        right_idx = tf.searchsorted(
            broadcast_sender_ids, flat_node_ids, side="right")

        # (P) -> (P, 1, 1) -> (P, B, K)
        prefix_ids = tf.math.add(
            tf.zeros_like(flat_node_ids), expand(expand(tf.range(prefix_size)))
        )
        # (B) -> (B, 1) -> (P, B, K)
        batch_ids = tf.math.add(
            tf.zeros_like(flat_node_ids), expand(tf.range(self.batch_size))
        )
        # (P, B, K, 1 + 1 + 1)
        left_idx_3d = tf.concat([
            expand(prefix_ids), expand(batch_ids), expand(left_idx)
        ], axis=-1)
        right_idx_3d = tf.concat([
            expand(prefix_ids), expand(batch_ids), expand(right_idx)
        ], axis=-1)

        # Trick:
        #    sender_ids: [1, 2, 3, 4, 4, 4, 5, 6]
        #    node_ids:   [2, 3, 4]
        #    left_idx:   [1, 2, 3]
        #    right_idx:  [2, 3, 6]
        #    left_mark:  [0,  1,  1,  1,  0,  0,  0,  0]
        #    right_mark: [0,  0, -1, -1,  0,  0, -1,  0]
        #    left+right: [0,  1,  0,  0,  0,  0, -1,  0]
        #    cumsum:     [0,  1,  1,  1,  1,  1,  0,  0]

        # scatter -> (P, B, E)
        flat_mask_shape = tf.stack([
            prefix_size, self.batch_size, (self.max_num_edges + 1)
        ])
        left_mark = tf.scatter_nd(
            indices=left_idx_3d,
            updates=tf.ones_like(flat_node_ids, dtype=tf.int32),
            shape=flat_mask_shape
        )
        right_mark = tf.scatter_nd(
            indices=right_idx_3d,
            updates=tf.negative(tf.ones_like(flat_node_ids, dtype=tf.int32)),
            shape=flat_mask_shape
        )
        flat_mask = tf.math.cumsum(tf.math.add(left_mark, right_mark), axis=-1)
        flat_mask = flat_mask[..., :-1]
        mask = tf.reshape(flat_mask, shape=tf.stack([
            *tf.unstack(prefix_shape), self.batch_size, self.max_num_edges
        ]))
        return tf.math.not_equal(mask, tf.constant(0, dtype=tf.int32))

    def reduce_sum_nodal(self, nodal_metrics, scale=False):
        '''
        Args:
          metrics: A (..., B, N) Tensor.

        Returns:
          reduced: A (..., B) Tensor.
        '''
        masked_metrics = tf.math.multiply(nodal_metrics, self.node_mask)
        if scale:
            return tf.math.multiply(
                tf.math.reduce_sum(masked_metrics, axis=-1),
                tf.math.divide(self.max_num_nodes, self.num_nodes)
            )
        return tf.math.reduce_sum(masked_metrics, axis=-1)

    def batch_avg(self, batch_metrics):
        '''
        Args:
          batch_metrics: A (..., B) Tensor.

        Returns:
          reduced: A (..., B) Tensor.
        '''
        return tf.math.divide(
            batch_metrics, tf.cast(self.num_nodes, tf.float32)
        )

    def _expand_last_dims(self, x, k):
        for _ in range(k):
            x = tf.expand_dims(x, axis=-1)
        return x

    def mask_nodal_info(self, nodal_info, ndims=1):
        return tf.math.multiply(
            nodal_info, self._expand_last_dims(self.node_mask, ndims)
        )

    def mask_edge_info(self, edge_info, ndims=1):
        return tf.math.multiply(
            edge_info, self._expand_last_dims(self.edge_mask, ndims)
        )

    def reverse(self):
        if self._reversed is not None:
            return self._reversed

        max_num_edges = self.max_num_edges

        def fn(params):
            edges, num_nodes, num_edges = params
            true_edges = edges[:num_edges, :]
            # (E, 2) and (E)
            reverse, indices = tf_lexsort(tf.reverse(true_edges, axis=[-1]))
            padded_reverse = tf_pad_axis_to(reverse, -2, max_num_edges)
            padded_indices = tf_pad_axis_to(indices, -1, max_num_edges)
            return padded_reverse, padded_indices

        reverse_edges, reverse_indices = tf.map_fn(
            fn, (self.edges, self.num_nodes, self.num_edges),
            dtype=(tf.int32, tf.int32)
        )
        reverse_edge_attrs = None
        if self.edge_attrs is not None:
            reverse_edge_attrs = self.mask_edge_info(
                tf.batch_gather(self.edge_attrs, reverse_indices), ndims=1
            )
        return BaseRuntimeGraph(
            edges=reverse_edges,
            node_mask=tf.cast(self.node_mask, tf.int32),
            center_mask=tf.cast(self.center_mask, tf.int32),
            edge_mask=tf.cast(self.edge_mask, tf.int32),
            dense=(self.dense_adjacency is not None),
            node_attrs=self.node_attrs,
            edge_attrs=reverse_edge_attrs,
            reversed=self
        )


class RuntimeGraph(BaseRuntimeGraph):
    def __init__(self, edges, center_mask, node_mask, edge_mask,
                 dense=False, node_attrs=None, edge_attrs=None):
        super(RuntimeGraph, self).__init__(
            edges=edges, center_mask=center_mask,
            node_mask=node_mask, edge_mask=edge_mask,
            dense=dense, node_attrs=node_attrs, edge_attrs=edge_attrs
        )
        self.__reversed = super(RuntimeGraph, self).reverse()

    def reverse(self):
        return self.__reversed
