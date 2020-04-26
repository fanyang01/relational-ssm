#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from load import MultiDataset, batched_input_fn_wrapper

from absl import app
from absl import flags

import os
import scipy
import numpy as np


FLAGS = flags.FLAGS


flags.DEFINE_string("to", default="nri", help="NRI or MA.")
flags.DEFINE_string(
    "dataset", default=os.path.join("datasets", "toy-1000"),
    help="Directory where data is stored.")


def nri():
    dstdir = FLAGS.dataset + "_nri_format"
    os.mkdir(dstdir)
    for subdir in ["train", "eval", "test"]:
        path = os.path.join(FLAGS.dataset, subdir)
        observations, adjs = convert(path)
        name = subdir if subdir != "eval" else "valid"
        np.save(
            os.path.join(dstdir, "feat_{}_gen.npy".format(name)),
            observations
        )
        np.save(
            os.path.join(dstdir, "edges_{}_gen.npy".format(name)),
            adjs
        )


def ma():
    dstdir = FLAGS.dataset + "_ma_format"
    os.mkdir(dstdir)
    dstdir = os.path.join(dstdir, "data")
    os.mkdir(dstdir)
    macro_path = os.path.join(dstdir, "macro_intents")
    os.mkdir(macro_path)

    for subdir in ["train", "test"]:
        path = os.path.join(FLAGS.dataset, subdir)
        name = subdir

        observations, _ = convert(path)
        # (B, N, T, dx) -> (B, T, N, dx)
        # (0, 1, 2, 3) -> (0, 2, 1, 3)
        observations = np.transpose(observations, [0, 2, 1, 3])
        shape = observations.shape
        assert len(shape) == 4
        # (B, T, N, dx) -> (B, T, N * dx)
        observations = np.reshape(observations, [shape[0], shape[1], -1])
        print(observations.shape)
        np.savez(
            os.path.join(dstdir, "gen_{}.npz".format(name)),
            data=observations
        )
        np.savez(
            os.path.join(macro_path, "gen_{}_friendly.npz".format(name)),
            data=np.zeros([shape[0], shape[1], 1])
        )


def main(argv):
    if FLAGS.to == "nri":
        nri()
    elif FLAGS.to == "ma":
        ma()
    else:
        raise ValueError("unknown format: " + FLAGS.to)


def convert(path):
    dataset = MultiDataset(path)
    fn = batched_input_fn_wrapper(
        dataset.build_full_input_fn(), dataset.num_datasets
    )
    batches, EOF = fn()
    assert EOF

    observations = batches.observations
    num_vertices = observations.shape[-2]

    # (T, B, N, dx) -> (B, N, T, dx)
    # (0, 1, 2, 3) -> (1, 2, 0, 3)
    observations = np.transpose(observations, [1, 2, 0, 3])

    adjs = []
    for i in range(dataset.num_datasets):
        edges = batches.edges[i]
        adj = scipy.sparse.coo_matrix(
            (
                np.ones(np.shape(edges)[0], dtype=np.int8),
                (edges[:, 0], edges[:, 1])
            ), shape=(num_vertices, num_vertices)
        )
        adjs.append(adj.toarray())

    adjs = np.stack(adjs, axis=0)  # (B, N, N)
    return observations, adjs


if __name__ == "__main__":
    app.run(main)
