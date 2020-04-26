from __future__ import absolute_import, division, print_function

from datautil import ZscoreTransform, ExampleContainer

import configparser
import os
import csv
import random
import functools

import numpy as np


def read_csv_as_array_slow(dir, file, skip_header=False, dtype=np.float32):
    path = os.path.join(dir, file)
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)  # skip the header
        for line in reader:
            attrs = [float(col) for col in line]
            rows.append(np.array(attrs, dtype=dtype))
    return np.stack(rows)


def read_csv_as_array(dir, file, skip_header=False, dtype=np.float32):
    path = os.path.join(dir, file)
    if not os.path.exists(path):
        return None
    return np.loadtxt(
        path, delimiter=',', dtype=dtype,
        skiprows=(1 if skip_header else 0)
    )


def load_example(dir):
    '''Load a example from `dir`.

    There should be at least two files under `dir`:

    - `properties.ini`, Required, which describes the basic properties of
      the example. It must contain two keys `NumDims`
      and `NumNodes` under the `DATASET` section.
    - `data.csv`, Required, which contains `NumNodes*NumDims` lines.
      Each line is a univariate time series.
    - `graph.csv`, Optional, which specifies all edges of the graph.
      The content must be a CSV header `FROM,TO` followed
      by a set of records, formatted as `SID,RID`.
    - `node-attrs.csv`, Optional, which contains `NumNodes` lines.
    - `edge-attrs.csv`, Optional, which contains the same #lines
      as `graph.csv`.
    - `time-attrs.csv`, Optional, which contains `T` lines.

    Returns:
        observations: A (N, T, d) numpy array.
        edges: A (E, 2) numpy array.
        extra_features: A 3-tuple of:
          - node_attrs: A (N, d0) numpy array.
          - edge_attrs: A (E, d1) numpy array.
          - time_attrs: A (T, d2) numpy array.
    '''
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(os.path.join(dir, "properties.ini"))
    config = config["DATASET"]
    num_nodes, num_dims = int(config["NumNodes"]), int(config["NumDims"])

    read_csv = functools.partial(read_csv_as_array, dir)

    flatten_data = read_csv("data.csv", skip_header=False)
    assert flatten_data.shape[0] == (num_nodes * num_dims)
    num_time_steps = flatten_data.shape[1]
    observations = np.split(flatten_data, num_nodes, axis=0)
    observations = np.transpose(
        np.stack(observations, axis=0), [0, 2, 1]
    )  # (N, d, T) -> (N, T, d)
    shape = observations.shape
    assert np.array_equal([shape[0], shape[2]], [num_nodes, num_dims])

    edges = read_csv("graph.csv", skip_header=True, dtype=np.int32)
    if edges is not None:
        assert edges.shape[1] == 2
        assert np.min(edges) >= 0 and np.max(edges) < num_nodes
    else:
        edges = np.stack([
            np.arange(num_nodes, dtype=np.int32)
        ] * 2, axis=-1)
    num_edges = edges.shape[0]

    node_attrs = read_csv("node-attrs.csv", skip_header=True)
    if node_attrs is not None:
        assert node_attrs.shape[0] == num_nodes
        assert np.array_equal(
            np.arange(num_nodes, dtype=np.float32), node_attrs[:, 0]
        )
        node_attrs = node_attrs[:, 1:]

    edge_attrs = read_csv("edge-attrs.csv", skip_header=True)
    if edge_attrs is not None:
        assert edge_attrs.shape[0] == num_edges
        assert np.array_equal(edges.astype(np.float32), edge_attrs[:, 0:2])
        edge_attrs = edge_attrs[:, 2:]

    time_attrs = read_csv("time-attrs.csv", skip_header=True)
    if time_attrs is not None:
        time_attrs = np.transpose(time_attrs)
        assert time_attrs.shape[0] == num_time_steps

    extra_features = (node_attrs, edge_attrs, time_attrs)
    return observations, edges, extra_features


class DirLoader(ExampleContainer):
    def __init__(self, dir, zscore=False, transform=None):
        if zscore and transform is not None:
            raise ValueError("Incorrect normalization.")

        observations, edges, extra = load_example(dir)
        if zscore:
            print("Normalize the dataset {} into Z-score...".format(dir))
            transform = ZscoreTransform.initialize(observations)

        super(DirLoader, self).__init__(
            observations, edges,
            transform=transform, extra_features=extra
        )


class FakeLoader(ExampleContainer):
    def __init__(self):
        """Builds fake data for testing."""
        num_nodes = 30
        dim_observ, num_steps = 3, 60
        shape = [num_steps, num_nodes, dim_observ]
        adjacency = np.eye(num_nodes)
        adjacency += np.random.randint(0, 2, size=adjacency.shape)
        edges = np.transpose(np.where(np.not_equal(adjacency, 0)))
        observations = np.random.rand(*shape).astype("float32")
        super(FakeLoader, self).__init__(observations, edges)


class MultiDirLoader(object):
    def __init__(self, root, zscore=False, transform=None):
        subdirs = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ]
        if len(subdirs) < 1:
            raise ValueError("#datasets = 0")
        random.shuffle(subdirs)

        if zscore and transform is not None:
            raise ValueError("Incorrect normalization.")
        elif zscore:
            S = min(100, len(subdirs))
            flatten_list = []
            for subdir in subdirs[:S]:
                observations, _, _ = load_example(subdir)
                dim_observ = observations.shape[-1]
                flat_observations = np.reshape(observations, (-1, dim_observ))
                flatten_list.append(flat_observations)
            flatten = np.concatenate(flatten_list, axis=0)
            transform = ZscoreTransform.initialize(flatten)

        container = DirLoader(subdirs[0])
        self._transform = transform
        self._subdirs = subdirs
        self._dim_observs = container.dim_observs
        self._dim_node_attrs = container.dim_node_attrs
        self._dim_edge_attrs = container.dim_edge_attrs
        self._dim_time_attrs = container.dim_time_attrs
        self._idx = 0

    @property
    def dim_observs(self):
        return self._dim_observs

    @property
    def dim_node_attrs(self):
        return self._dim_node_attrs

    @property
    def dim_edge_attrs(self):
        return self._dim_edge_attrs

    @property
    def dim_time_attrs(self):
        return self._dim_time_attrs

    @property
    def transform(self):
        return self._transform

    @property
    def num_examples(self):
        return len(self._subdirs)

    def access(self, idx):
        subdir = self._subdirs[idx]
        print("Loading new example {} ...".format(subdir))
        container = DirLoader(subdir, zscore=False, transform=self._transform)
        if container.dim_observs != self.dim_observs:
            raise ValueError("Incompatible example: " + subdir)
        return container

    def _next_example(self):
        container = self.access(self._idx)
        EOF = False
        self._idx += 1
        if self._idx >= len(self._subdirs):
            EOF = True
            random.shuffle(self._subdirs)
            self._idx = 0

        return container, EOF

    def next_example_fn(self):
        def _fn(*unused_args):
            container, EOF = self._next_example()
            example, _ = container.next_example_fn()()
            return example, EOF
        return _fn
