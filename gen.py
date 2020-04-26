#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import csv
import shutil
import configparser
from absl import flags
import numpy as np
from scipy import stats

flags.DEFINE_string(
    "graph", default="sbm", help="random graph model to use (indep|sbm)")
flags.DEFINE_integer(
    "train_nodes", default=36, help="Number of nodes in training dataset.")
flags.DEFINE_integer(
    "eval_nodes", default=36, help="Number of nodes in evaluation dataset.")
flags.DEFINE_integer(
    "test_nodes", default=36, help="Number of nodes in test dataset.")
flags.DEFINE_string(
    "dataset", default="test", help="Name of the dataset.")
flags.DEFINE_bool(
    "multi", default=False, help="Generate multiple datasets.")
flags.DEFINE_integer(
    "num_datasets", default=1, help="Number of datasets to generate.")
flags.DEFINE_integer(
    "dims", default=1,
    help="Number of dimensions in the observations (x_t).")
flags.DEFINE_integer(
    "steps", default=80, help="Number of time steps.")

flags.DEFINE_integer(
    "dim_node_vec", default=4,
    help="Number of dimensions in node-specific random vectors.")
flags.DEFINE_float(
    "beta_0", default=0.0,
    help="The constant in NVAR model.")
flags.DEFINE_float(
    "beta_1", default=5.0,
    help="The coefficient of neighbor impact in NVAR model.")
flags.DEFINE_float(
    "beta_2", default=-1.5,
    help="The coefficient of self impact in NVAR model.")
flags.DEFINE_float(
    "z_noise_scale", default=0.05,
    help="The stddev of white noise of latent states.")
flags.DEFINE_float(
    "x_noise_scale", default=0.05,
    help="The stddev of white noise of observations.")
flags.DEFINE_list(
    "eta", default=[-1.5, 0.4, 2.0, -0.9],
    help="nodal impact cofficients.")
flags.DEFINE_list(
    "psi", default=[2.5],
    help="Observation multipliers.")
flags.DEFINE_bool(
    "mul_noise", default=False,
    help="Add multiplicative noise rather than additive noise.")

flags.DEFINE_float(
    "indep_c1", default=20.0, help="P[E_ij = (1,1)] = c1 / N.")
flags.DEFINE_float(
    "indep_c2", default=0.8,
    help="P[E_ij = (0,1)] = P[E_ij = (1,0)] = 0.5 / (N ^ c2).")

flags.DEFINE_float(
    "sbm_k", default=3, help="Number of blocks in SBM.")
flags.DEFINE_float(
    "sbm_c1", default=0.5,
    help="P[E_ij = 1 | Ci = Cj] = 2.0 / (N ^ c1)")
flags.DEFINE_float(
    "sbm_c2", default=1.0,
    help="P[E_ij = 1 | Ci != Cj] = 2.0 / (N ^ c2)")

flags.DEFINE_bool(
    "norm", default=False, help="Perform Z-normalization.")
flags.DEFINE_bool(
    "overwrite", default=False,
    help="If true, deletes existing `dataset` directory.")

FLAGS = flags.FLAGS


def gen_indep_adjacency(N, c1=20.0, c2=0.8):
    p_11 = c1 / N
    p_01 = 0.5 / (N ** c2)
    p_10 = p_01
    adj = np.eye(N, dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            rand = np.random.rand()
            if rand < p_11:
                adj[i, j], adj[j, i] = 1, 1
            elif rand < p_11 + p_01:
                adj[i, j], adj[j, i] = 0, 1
            elif rand < p_11 + p_01 + p_10:
                adj[i, j], adj[j, i] = 1, 0
            else:
                adj[i, j], adj[j, i] = 0, 0
    return adj


def gen_sbm_adjacency(N, K=5, c_intra=0.5, c_inter=1.0):
    p_intra = 2.0 / (N ** c_intra)
    p_inter = 2.0 / (N ** c_inter)
    labels = np.random.randint(0, K, size=N)
    adj = np.eye(N, dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            rand = np.random.rand()
            if labels[i] == labels[j]:
                adj[i, j] = 1 if rand < p_intra else 0
            else:
                adj[i, j] = 1 if rand < p_inter else 0
            adj[j, i] = adj[i, j]
    return adj


def gen_latent_states(adj, num_steps, config):
    '''
    Args:
      adj: A (N, N) ndarray.

    Returns:
      states: A (T, N, 1) ndarray.
    '''
    N = np.shape(adj)[0]
    dh = config.dim_node_vec

    beta_0 = config.beta_0

    states = [np.zeros(N)]
    node_vectors = np.random.normal(size=[N, dh])
    if config.eta is None:
        node_cofficients = np.random.normal(size=dh)
    elif len(config.eta) == dh:
        node_cofficients = np.array([float(i) for i in config.eta])
    else:
        raise ValueError("len(eta) != dm")

    beta_1 = config.beta_1
    # indegrees = np.sum(adj, axis=0)
    outdegrees = np.sum(adj, axis=1)

    beta_2 = config.beta_2

    for t in range(num_steps):
        nodal_impact = beta_0 + np.dot(node_vectors, node_cofficients)

        prev_state = states[t]
        # The NVAR paper said node i is likely to be affected by node j
        # if i follows j. So it is the reverse version of graph diffusion
        # or message passing.
        #
        # neighbor_impact = np.matmul(np.transpose(adj), prev_state)
        # neighbor_impact = np.divide(neighbor_impact, indegrees)
        #
        neighbor_impact = np.matmul(adj, prev_state)
        neighbor_impact = np.divide(neighbor_impact, outdegrees)
        neighbor_impact = beta_1 * neighbor_impact

        self_impact = beta_2 * prev_state

        noise = np.random.normal(
            loc=0.0, scale=config.z_noise_scale, size=N)

        new_state = nodal_impact + neighbor_impact + self_impact
        new_state = np.cos(new_state) + noise
        states.append(new_state)
    return np.expand_dims(np.array(states[1:]), -1)


def gen_observations(latent_states, measure_matrix, noise_scale,
                     mul_noise=False):
    '''
    Args:
      latent_states: A (T, N, dz) ndarray.

    Returns:
      observations: A (T, N, dx) ndarray.
    '''
    assert np.size(np.shape(latent_states)) == 3
    assert np.size(np.shape(measure_matrix)) == 2
    measurements = np.tanh(np.matmul(latent_states, measure_matrix))
    noise = np.random.normal(
        loc=0.0, scale=noise_scale, size=np.shape(measurements)
    )
    if mul_noise:
        return np.multiply(measurements, noise)
    return np.add(measurements, noise)


def gen_dataset(dir, num_nodes, measure_matrix, num_steps, config):
    files = ["data.csv", "graph.csv", "properties.ini"]
    datafile, graphfile, propertyfile = tuple(
        [os.path.join(dir, f) for f in files]
    )

    if config.graph == "sbm":
        adj = gen_sbm_adjacency(
            N=num_nodes, K=config.sbm_k,
            c_intra=config.sbm_c1,
            c_inter=config.sbm_c2
        )
    elif config.graph == "indep":
        adj = gen_indep_adjacency(
            N=num_nodes,
            c1=config.indep_c1,
            c2=config.indep_c2
        )
    else:
        raise ValueError("unknown random graph model: " + config.graph)

    latent_states = gen_latent_states(adj, num_steps, config)
    observs = gen_observations(
        latent_states, measure_matrix, config.x_noise_scale,
        mul_noise=config.mul_noise
    )
    num_dims = np.shape(measure_matrix)[-1]

    # (T, N, d) -> (N, d, T) -> (N*d, T)
    observs = np.reshape(
        np.transpose(observs, [1, 2, 0]),
        [num_nodes * num_dims, num_steps]
    )
    if config.norm:
        observs = stats.zscore(observs, axis=None)

    print("Write {} ...".format(datafile))
    np.savetxt(datafile, observs, fmt="%.4f", delimiter=",")

    print("Write {} ...".format(graphfile))
    with open(graphfile, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["FROM", "TO"])
        indicies = np.transpose(np.nonzero(adj))
        for e in indicies:
            writer.writerow(e.tolist())

    properties = configparser.ConfigParser()
    properties.optionxform = str
    properties['DATASET'] = {
        'NumNodes': num_nodes,
        'NumDims': num_dims,
        'MeasureMatrix': np.array2string(measure_matrix, precision=3),
        'RandGraphModel': config.graph,
        'StateNoise': config.z_noise_scale,
        'ObservNoise': config.x_noise_scale,
        'SBM_NumBlocks': config.sbm_k,
        'SBM_C1': config.sbm_c1,
        'SBM_C2': config.sbm_c2,
        'Indep_C1': config.indep_c1,
        'Indep_C2': config.indep_c2,
        'Norm': config.norm
    }
    print("Write {} ...".format(propertyfile))
    with open(propertyfile, "w") as f:
        properties.write(f)


def main(argv):
    FLAGS(argv)

    dir = os.path.join("datasets", FLAGS.dataset)
    if os.path.exists(dir):
        if not FLAGS.overwrite:
            print("Error: {} already exists.".format(dir))
            sys.exit(1)
        shutil.rmtree(dir)
    os.makedirs(dir)

    num_dims, num_steps = FLAGS.dims, FLAGS.steps
    num_train_nodes, num_eval_nodes, num_test_nodes = \
        FLAGS.train_nodes, FLAGS.eval_nodes, FLAGS.test_nodes

    dz, dx = 1, num_dims
    if FLAGS.psi is None:
        measure_matrix = np.random.normal(loc=2.0, scale=1.0, size=[dz, dx])
    elif len(FLAGS.psi) == dx:
        measure_matrix = np.array([float(i) for i in FLAGS.psi])
        measure_matrix = np.expand_dims(measure_matrix, axis=0)
    else:
        raise ValueError("len(psi) != dx")

    assert not (FLAGS.num_datasets > 1 and not FLAGS.multi)

    def gen_and_write(subdir, num_nodes):
        dataset_dir = os.path.join(dir, subdir)
        if not FLAGS.multi:
            os.mkdir(dataset_dir)
            gen_dataset(
                dataset_dir, num_nodes, measure_matrix, num_steps, FLAGS
            )
            return

        os.mkdir(dataset_dir)
        for i in range(FLAGS.num_datasets):
            dataset_subdir = os.path.join(dataset_dir, str(i))
            os.mkdir(dataset_subdir)
            gen_dataset(
                dataset_subdir, num_nodes, measure_matrix, num_steps, FLAGS
            )

    gen_and_write("train", num_train_nodes)
    gen_and_write("eval", num_eval_nodes)
    gen_and_write("test", num_test_nodes)

    print("Done.")


if __name__ == "__main__":
    main(sys.argv)
