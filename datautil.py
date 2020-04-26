from __future__ import absolute_import, division, print_function

import math

import numpy as np

import graph


class ZscoreTransform(object):
    @staticmethod
    def initialize(observations):
        shape = observations.shape
        flattened = np.reshape(observations, (-1, shape[-1]))
        mean = np.mean(flattened, axis=0)
        stddev = np.std(flattened, axis=0)
        return ZscoreTransform(mean, stddev)

    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def _arr2str(self, x):
        return np.array2string(x, precision=4, suppress_small=True)

    def forward_prompt(self):
        return "Apply forward transformation (A - %s) / %s ..." \
            % (self._arr2str(self.mean), self._arr2str(self.stddev))

    def inverse_prompt(self):
        return "Apply inverse transformation (A * %s + %s ..." \
            % (self._arr2str(self.mean), self._arr2str(self.stddev))

    def forward(self, x):
        return np.divide(np.subtract(x, self.mean), self.stddev)

    def inverse(self, x):
        return np.add(np.multiply(x, self.stddev), self.mean)


class ExampleContainer(object):
    def __init__(self, observations, edges,
                 transform=None, extra_features=None):
        '''
        Args:
          observations: A (N, T, d) numpy array.
          edges: A (E, 2) numpy array.
          transform: Optional, a ZscoreTransform object.
          extra_features: Optional, a 3-tuple of:
            - node_attrs: A (N, d0) numpy array.
            - edge_attrs: A (E, d1) numpy array.
            - time_attrs: A (T, d2) numpy array.
        '''
        if transform is not None:
            observations = transform.forward(observations)
        self._observations = observations  # (N, T, d)
        self._edges = edges  # (E, 2)
        self._transform = transform
        self._extra_features = extra_features

        if extra_features is not None:
            self._node_attrs = extra_features[0]
            self._edge_attrs = extra_features[1]
            self._time_attrs = extra_features[2]

    @property
    def observations(self):
        return self._observations

    @property
    def edges(self):
        return self._edges

    @property
    def extra_features(self):
        return self._extra_features

    @property
    def transform(self):
        return self._transform

    @property
    def num_vertices(self):
        return self.observations.shape[0]

    @property
    def num_time_steps(self):
        return self.observations.shape[1]

    @property
    def dim_observs(self):
        return self.observations.shape[2]

    @property
    def dim_node_attrs(self):
        if self._node_attrs is not None:
            return self._node_attrs.shape[-1]
        return 0

    @property
    def dim_edge_attrs(self):
        if self._edge_attrs is not None:
            return self._edge_attrs.shape[-1]
        return 0

    @property
    def dim_time_attrs(self):
        if self._time_attrs is not None:
            return self._time_attrs.shape[-1]
        return 0

    def next_example_fn(self):
        # (N, T, d) -> (T, N, d)
        observations = np.transpose(self.observations, [1, 0, 2])
        EOF = True

        def _fn(*unused_args):
            return Example(
                observations=observations,
                edges=self.edges,
                center_mask=np.ones(self.num_vertices),
                node_attrs=self._node_attrs,
                edge_attrs=self._edge_attrs,
                time_attrs=self._time_attrs
            ), EOF

        return _fn

    def split(self, fraction=0.5):
        split_time_steps = int(math.ceil(self.num_time_steps * fraction))
        split_0 = self.observations[:, split_time_steps:, :]
        split_1 = self.observations[:, :split_time_steps, :]

        extra_features_0, extra_features_1 = None, None
        if self.dim_time_attrs > 0:
            extra_features_0 = (
                self._node_attrs, self._edge_attrs,
                self._time_attrs[split_time_steps:]
            )
            extra_features_1 = (
                self._node_attrs, self._edge_attrs,
                self._time_attrs[:split_time_steps]
            )

        container_0 = ExampleContainer(
            observations=split_0, edges=self.edges,
            extra_features=extra_features_0,
            transform=self.transform
        )
        container_1 = ExampleContainer(
            observations=split_1, edges=self.edges,
            extra_features=extra_features_1,
            transform=self.transform
        )
        return container_0, container_1


class AbstractExample(object):
    def __init__(self, observations, edges, center_mask,
                 reverse_edges, reverse_indices,
                 node_attrs=None, edge_attrs=None, time_attrs=None):
        self._observations = observations         # (..., T, N, dx)
        self._edges = edges                       # (..., E, 2)
        self._center_mask = center_mask           # (..., N)
        self._reverse_edges = reverse_edges       # (..., E, 2)
        self._reverse_indices = reverse_indices   # (..., E)
        self._node_attrs = node_attrs             # (..., N, dv)
        self._edge_attrs = edge_attrs             # (..., E, de)
        self._time_attrs = time_attrs             # (..., T, du)

    @property
    def num_time_steps(self):
        return self._observations.shape[0]

    @property
    def dim_observs(self):
        return self._observations.shape[-1]

    @property
    def num_nodes(self):
        return self._observations.shape[-2]

    @property
    def num_edges(self):
        return self._edges.shape[0]

    @property
    def observations(self):
        ''' A (T, N, d) array. '''
        return self._observations

    @observations.setter
    def observations(self, value):
        self._observations = value

    @property
    def edges(self):
        return self._edges

    @property
    def reverse_edges(self):
        return self._reverse_edges

    @property
    def reverse_indices(self):
        return self._reverse_indices

    @property
    def center_mask(self):
        return self._center_mask

    @property
    def node_attrs(self):
        return self._node_attrs

    @property
    def edge_attrs(self):
        return self._edge_attrs

    @property
    def time_attrs(self):
        return self._time_attrs

    @time_attrs.setter
    def time_attrs(self, value):
        self._time_attrs = value


class Example(AbstractExample):
    def __init__(self, observations, edges, *args, **kwargs):
        reverse, indices = graph.lexsort(np.flip(edges, axis=-1))
        super(Example, self).__init__(
            observations, edges, *args,
            reverse_edges=reverse, reverse_indices=indices,
            **kwargs
        )

    def clone(self):
        return Example(
            observations=self.observations,
            edges=self.edges,
            center_mask=self.center_mask,
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            time_attrs=self.time_attrs
        )

    def slice(self, ti, tj):
        mini_batch = self.clone()
        mini_batch.observations = self.observations[ti:tj]
        if self.time_attrs is not None:
            mini_batch.time_attrs = self.time_attrs[ti:tj]
        return mini_batch


class Batch(AbstractExample):
    def __init__(self, *args, node_mask, edge_mask, **kwargs):
        super(Batch, self).__init__(*args, **kwargs)
        self._node_mask = node_mask
        self._edge_mask = edge_mask

    @property
    def node_mask(self):
        return self._node_mask

    @property
    def edge_mask(self):
        return self._edge_mask

    @property
    def num_nodes(self):
        return np.sum(self._node_mask, axis=-1)

    @property
    def num_edges(self):
        return np.sum(self._edge_mask, axis=-1)

    def clone(self):
        return Batch(
            observations=self.observations,
            edges=self.edges,
            reverse_edges=self.reverse_edges,
            reverse_indices=self.reverse_indices,
            center_mask=self.center_mask,
            node_mask=self.node_mask,
            edge_mask=self.edge_mask,
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            time_attrs=self.time_attrs
        )

    def slice(self, ti, tj):
        mini_batch = self.clone()
        mini_batch.observations = self.observations[ti:tj]
        if self.time_attrs is not None:
            mini_batch.time_attrs = self.time_attrs[ti:tj]
        return mini_batch


def pad_axis_to(a, axis, size, const=None):
    shape = np.copy(a.shape)
    if shape[axis] > size:
        raise ValueError("shape[axis] > size")
    elif shape[axis] == size:
        return a
    shape[axis] = size - shape[axis]
    if const is not None:
        return np.concatenate([a, np.full(shape, const)], axis=axis)
    return np.concatenate([a, np.zeros(shape, dtype=a.dtype)], axis=axis)


def gen_padding_masks(sizes, max_size):
    pads = max_size - sizes
    return np.stack([
        np.concatenate([np.ones(m), np.zeros(n)])
        for m, n in zip(sizes, pads)
    ])


def batch_and_mask(examples):
    assert type(examples[0]) is Example
    batched_time_attrs = batched_node_attrs = batched_edge_attrs = None

    num_steps_per_batch = [batch.num_time_steps for batch in examples]
    assert np.array_equal(
        np.divide(num_steps_per_batch, num_steps_per_batch[0]),
        np.ones_like(num_steps_per_batch)
    )

    if examples[0].time_attrs is not None:
        batched_time_attrs = np.stack(
            [batch.time_attrs for batch in examples], axis=-2
        )  # B * (T, d0) -> (T, B, d0)

    num_nodes = [batch.num_nodes for batch in examples]
    max_num_nodes = np.amax(num_nodes)
    node_masks = gen_padding_masks(num_nodes, max_num_nodes)  # (B, N)

    # B * (T, N, dx) -> (T, B, N, dx)
    batched_observations = np.stack([
        pad_axis_to(example.observations, -2, max_num_nodes)
        for example in examples
    ], axis=-3)
    assert np.size(batched_observations.shape) == 4

    # B * (N) -> (B, N)
    batched_center_mask = np.stack([
        pad_axis_to(example.center_mask, -1, max_num_nodes)
        for example in examples
    ], axis=0)

    # B * (N, d1) -> (B, N, d1)
    if examples[0].node_attrs is not None:
        batched_node_attrs = np.stack([
            pad_axis_to(example.node_attrs, -2, max_num_nodes)
            for example in examples
        ], axis=0)

    num_edges = [example.num_edges for example in examples]
    max_num_edges = np.amax(num_edges)
    edge_masks = gen_padding_masks(num_edges, max_num_edges)  # (B, E)

    # B * (E, 2) -> (B, E, 2)
    batched_edges = np.stack([
        pad_axis_to(example.edges, -2, max_num_edges)
        for example in examples
    ], axis=0)

    # B * (E, d2) -> (B, E, d2)
    if examples[0].edge_attrs is not None:
        batched_edge_attrs = np.stack([
            pad_axis_to(example.edge_attrs, -2, max_num_edges)
            for example in examples
        ], axis=0)

    # B * (E, 2) -> (B, E, 2)
    batched_reverse_edges = np.stack([
        pad_axis_to(example.reverse_edges, -2, max_num_edges)
        for example in examples
    ], axis=0)

    # B * (E) -> (B, E)
    batched_reverse_indices = np.stack([
        pad_axis_to(example.reverse_indices, -1, max_num_edges)
        for example in examples
    ], axis=0)

    return Batch(
        observations=batched_observations,
        edges=batched_edges,
        reverse_edges=batched_reverse_edges,
        reverse_indices=batched_reverse_indices,
        center_mask=batched_center_mask,
        node_mask=node_masks,
        edge_mask=edge_masks,
        node_attrs=batched_node_attrs,
        edge_attrs=batched_edge_attrs,
        time_attrs=batched_time_attrs
    )


def sliding_window_wrapper(input_fn, win_size, stride, enable=False):
    current_example, EOF, reset, start_idx = None, False, False, 0

    def log_new(example):
        print("New example: shape={}".format(np.shape(example.observations)))

    def new_example(*unused_args):
        '''
        Returns:
          example: An Example whose `observations` is a (W, N, dx) ndarray.
          EOF: A boolean indicating whether end of file has been reached.
          # reset: A boolean indicating whether `input_fn` is called.
        '''
        nonlocal current_example, EOF, reset, start_idx

        if current_example is None or not enable:
            current_example, EOF = input_fn(*unused_args)
            reset = True
            log_new(current_example)
        if not enable:
            return current_example, EOF, reset

        end_idx = start_idx + win_size
        if end_idx > np.shape(current_example.observations)[0]:
            current_example, EOF = input_fn(*unused_args)
            start_idx, reset = 0, True
            end_idx = start_idx + win_size
            # log_new(current_example)

        example = current_example.slice(start_idx, end_idx)
        # print("Window [{}:{}]".format(start_idx, end_idx))

        start_idx += stride

        EOB = (start_idx + win_size) > \
            np.shape(current_example.observations)[0]

        # return example, (EOF and EOB), reset
        return example, (EOF and EOB)

    return new_example


def sliding_window_view(sequence, win_size):
    '''
    Args:
      sequence: A (T, ...) Tensor.
      win_size: An integer.

    Returns:
      stacked: A (T-W+1, W, ...) Tensor.
    '''
    T = np.shape(sequence)[0]
    windows = [sequence[t:t + win_size] for t in range(0, T - win_size + 1)]
    stacked = np.stack(windows, axis=0)
    return stacked


def random_window_wrapper(input_fn, win_size_lo, win_size_hi):
    '''
    Args:
      input_fn: `observations`: A (T, B, N, dx) ndarray.

    Returns:
      input_fn: A function.
    '''
    example, EOF = input_fn(0)
    if not EOF:
        print("Random window wrapper calls the inner `input_fn` only once.")
    T = example.num_time_steps
    assert 0 < win_size_lo and \
        win_size_lo <= win_size_hi and win_size_hi <= T

    def new_example(*unused_args):
        EOF = False
        rand_win_size = np.random.randint(win_size_lo, win_size_hi + 1)
        start_idx = np.random.randint(0, T - rand_win_size + 1)
        end_idx = start_idx + rand_win_size
        print("Window [{}:{}]".format(start_idx, end_idx))
        mini_example = example.slice(start_idx, end_idx)
        return mini_example, EOF

    return new_example


def batched_input_fn_wrapper(input_fn, batch_size):
    '''
    Args:
      input_fn: `observations`: a (T, N, dx) Tensor.
      batch_size: A scalar.

    Returns:
      input_fn: `observations`: a (T, B, N, dx) Tensor.
    '''
    assert batch_size > 0

    def new_batch(*unused_args):
        examples, EOF = [], False
        for _ in range(batch_size):
            example, EOF = input_fn(*unused_args)
            examples.append(example)
            if EOF:
                break
        batch = batch_and_mask(examples)
        print("#Nodes in each graph: {}".format(batch.num_nodes))
        return batch, EOF

    return new_batch
