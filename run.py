#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from dataset import build_dataloader, SAME_GRAPH
from datautil import sliding_window_view
from model import build_shared_model, build_tf_graph
from model import FlowSources, PersistentStates
from model import SUMMARY_TRAIN, SUMMARY_EVAL
from model import SUMMARY_QUICK_PREDICT, SUMMARY_SLOW_PREDICT
from model import MODE_TRAIN, MODE_EVAL, MODE_SAMPLE
from model import MODE_QUICK_PREDICT, MODE_SLOW_PREDICT
from gnn import COMBINER_LSTM, GNN_SPARSE, MESSENGER_BINARY
from vsmc import MULTINOMIAL_RESAMPLING

import os
import json
from collections import deque
import itertools

import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from tensorflow.python.training.summary_io import SummaryWriterCache

flags = tf.app.flags

flags.DEFINE_integer(
    "seed", default=(1 << 31) - 1, help="Random seed.")
flags.DEFINE_bool(
    "test", default=False, help="Validation or test.")
flags.DEFINE_string(
    "optimizer", default=None, help="Optimizer to use: (Adam|SGD).")
flags.DEFINE_float(
    "adam_beta1", default=0.9, help="Adam: beta1.")
flags.DEFINE_float(
    "adam_beta2", default=0.999, help="Adam: beta2.")
flags.DEFINE_float(
    "adam_eps", default=1E-8, help="Adam: epsilon.")
flags.DEFINE_float(
    "learning_rate", default=0.01, help="Initial learning rate.")
flags.DEFINE_integer(
    "learning_rate_warmup_steps", default=1000,
    help="Number of steps to warm up the learning rate.")
flags.DEFINE_float(
    "learning_rate_min_ratio", default=0.001,
    help="Minimum learning rate as a percentage of `learning_rate`.")
flags.DEFINE_float(
    "learning_rate_init_variance", default=1.0,
    help="Initial variance for noisy linear cosine decay .")
flags.DEFINE_float(
    "learning_rate_variance_decay", default=0.66,
    help="Initial variance for noisy linear cosine decay .")
flags.DEFINE_integer(
    "decay_steps", default=None, help="Number of steps to decay over.")
flags.DEFINE_bool(
    "clip_gradient", default=False, help="If true, perform gradient clipping.")
flags.DEFINE_float(
    "clip_threshold", default=10.0, help="Threshold of global norm.")
flags.DEFINE_bool(
    "mini_batch", default=True, help="If true, use mini-batch training.")
flags.DEFINE_bool(
    "trace", default=False, help="If true, trace the runtime performance.")
flags.DEFINE_integer(
    "num_steps", default=500, help="Number of training steps to run.")
flags.DEFINE_bool(
    "reuse_batch", default=False,
    help="If true, reuse batches whose error are large.")
flags.DEFINE_integer(
    "err_win_size", default=10, help="Window size for smoothing losses.")
flags.DEFINE_integer(
    "reuse_batch_threshold", default=80,
    help="If current loss exceeds this threshold, reuse current mini-batch.")
flags.DEFINE_bool(
    "tbptt", default=False, help="If true, use truncated BPTT training.")
flags.DEFINE_integer(
    "tbptt_win_size", default=50, help="Window size for truncated BPTT.")
flags.DEFINE_string(
    "train_obj", default="iwae",
    help="Objective to use for training (vi|iwae|vsmc).")
flags.DEFINE_bool(
    "analytic_kl", default=True,
    help="Whether or not to use the analytic version of the KL.")
flags.DEFINE_bool(
    "kl_anneal", default=False,
    help="Whether or not to apply anneal factor to the KL.")
flags.DEFINE_integer(
    "train_batch_size", default=1, help="Batch size for training.")
flags.DEFINE_integer(
    "eval_batch_size", default=1, help="Batch size for evaluation.")
flags.DEFINE_integer(
    "const_num_nodes", default=None, help="Fixed number of nodes.")

flags.DEFINE_string(
    "vsmc_resample_impl", default=MULTINOMIAL_RESAMPLING,
    help="Resampling method to use in VSMC (multinomial|relaxed).")
flags.DEFINE_float(
    "vsmc_nasmc_weight", default=0.0,
    help="Weight of the NASMC term.")

flags.DEFINE_string(
    "aux_task", default=None, help="Auxiliary task to use (cpc|zf|adj|dgi).")
flags.DEFINE_float(
    "aux_weight", default=1.0, help="Weight of the auxiliary loss.")
flags.DEFINE_float(
    "aux_weight_min_ratio", default=0.1,
    help="Minimum weight as a percentage of `aux_weight`.")
flags.DEFINE_integer(
    "aux_weight_decay_steps", default=None,
    help="Number of steps to decay over.")
flags.DEFINE_float(
    "aux_cpc_scale", default=1.0, help="Weight of CPC scores in MIX.")
flags.DEFINE_string(
    "aux_cpc_state", default="z",
    help="Which state to use as context, (z|h).")
flags.DEFINE_float(
    "aux_dgi_scale", default=1.0, help="Weight of DGI scores in MIX.")
flags.DEFINE_float(
    "aux_zf_scale", default=0.1, help="Weight of Z-Forcing scores in MIX.")
flags.DEFINE_float(
    "aux_mask_scale", default=1.0, help="Weight of MD scores in MIX.")
flags.DEFINE_integer(
    "aux_zf_num_steps", default=5, help="Number of time steps to predict.")
flags.DEFINE_integer(
    "aux_mask_num_nodes", default=1, help="Number of nodes to mask.")
flags.DEFINE_bool(
    "aux_mask_all_at_once", default=True, help="Mask all nodes at once.")
flags.DEFINE_integer(
    "num_preview_steps", default=0,
    help="Number of steps to preview to encourage multi-step prediction.")
flags.DEFINE_float(
    "preview_loss_weight", default=0.5,
    help="Importance of multi-step previews for inference.")


flags.DEFINE_string(
    "proposal", default="joint",
    help="Proposal distribution to use: (indep|joint).")
flags.DEFINE_integer(
    "proposal_gnn_num_layers", default=1,
    help="Number of GNN layers in proposal model.")
flags.DEFINE_string(
    "proposal_loc_activation", default="linear",
    help="Activation function for proposed states.")
flags.DEFINE_bool(
    "proposal_loc_layer_norm", default=False,
    help="If true, apply layer normalization to loc.")
flags.DEFINE_bool(
    "proposal_scale_identical", default=False,
    help="If true, use identical variance.")
flags.DEFINE_bool(
    "denoising", default=True, help="Perturb the observations for proposal.")
flags.DEFINE_float(
    "noise_scale", default=0.1,
    help="Initial scale of Gaussian noise.")
flags.DEFINE_float(
    "noise_scale_min_ratio", default=0.01,
    help="Minimum noise scale as a percentage of `noise_scale`.")
flags.DEFINE_integer(
    "noise_scale_decay_steps", default=None,
    help="Number of steps to decay over.")
flags.DEFINE_bool(
    "use_belief", default=True, help="Use belief states in proposal.")
flags.DEFINE_bool(
    "use_lookahead", default=False, help="Use lookahead info in proposal.")
flags.DEFINE_bool(
    "use_skip_conn", default=True,
    help="Include prior locations in proposal design.")
flags.DEFINE_bool(
    "use_gated_adder", default=False,
    help="Include prior locations in proposal design.")
flags.DEFINE_bool(
    "stop_gen_gradient", default=False,
    help="Discard the gradient of gen parameters contributed by proposal.")
flags.DEFINE_bool(
    "interleaving", default=False,
    help="Interleave the bootstrap proposal and default proposal.")
flags.DEFINE_bool(
    "interleaving_randomly", default=True,
    help="Whether or not to randomize the interleaving process.")
flags.DEFINE_float(
    "interleaving_rate", default=1.0,
    help="Frequency to use the default proposal.")
flags.DEFINE_float(
    "interleaving_rate_min_ratio", default=0.1,
    help="Minimum interleaving rate as a percentage of `interleaving_rate`.")
flags.DEFINE_float(
    "interleaving_num_periods", default=0.5,
    help="Number of periods for warm restarts.")
flags.DEFINE_integer(
    "interleaving_decay_steps", default=None,
    help="Decay from `interleaving_rate` to " +
    "(`interleaving_rate` * `interleaving_rate_min_ratio`) " +
    "in first `interleaving_decay_steps`.")
flags.DEFINE_bool(
    "prefixing", default=False,
    help="Use bootstrap proposal in the suffix of sequence.")
flags.DEFINE_integer(
    "prefix_length", default=10000, help="Length of prefix.")
flags.DEFINE_integer(
    "num_samples", default=10,
    help="Number of samples to use in training.")
flags.DEFINE_integer(
    "eval_num_samples", default=10,
    help="Number of samples to use in evaluation.")

flags.DEFINE_bool(
    "his", default=False,
    help="Whether or not to use Hamiltonian importance sampling in SMC.")
flags.DEFINE_integer(
    "his_num_leapfrog_steps", default=10,
    help="Number of leapfrog steps performed by Hamiltonian IS.")
flags.DEFINE_float(
    "his_max_step_size", default=0.2,
    help="Maximum step size in Hamiltonian IS.")
flags.DEFINE_float(
    "his_mass_scale", default=1.0,
    help="Scale of the diagonal mass matrix used by Hamiltonian IS.")

flags.DEFINE_integer(
    "train_num_pred_steps", default=5,
    help="Number of time steps reserved for evaluating prediction accuracy.")
flags.DEFINE_integer(
    "train_num_pred_samples", default=10,
    help="Number of samples to use in prediction.")
flags.DEFINE_bool(
    "disable_eval", default=False, help="If true, disable evaluation.")
flags.DEFINE_integer(
    "eval_win_size", default=0,
    help="If > 0, partition the evaluation sequence into small windows.")
flags.DEFINE_integer(
    "eval_stride", default=20,
    help="Number of time steps to increment for next window.")
flags.DEFINE_integer(
    "eval_num_pred_steps", default=5,
    help="Number of time steps reserved for evaluating prediction accuracy.")
flags.DEFINE_integer(
    "eval_num_pred_samples", default=16,
    help="Number of samples to use in prediction.")


flags.DEFINE_integer(
    "parallel_iterations", default=10,
    help="Number of parallel iterations to run.")
flags.DEFINE_bool(
    "swap_memory", default=False,
    help="Enable swap GPU-CPU memory in while loops.")
flags.DEFINE_bool(
    "markov", default=False, help="Markov model or non-Markov model.")
flags.DEFINE_integer(
    "dim_latent", default=16,
    help="Number of dimensions in the latent states (z_t).")
flags.DEFINE_integer(
    "dim_hidden", default=64,
    help="Number of dimensions in the state summaries (h_t).")
flags.DEFINE_integer(
    "dim_mlp", default=64,
    help="Number of hidden units of two-layer MLPs.")

flags.DEFINE_bool(
    "learn_node_embed", default=False, help="Learn extra node embeddings.")
flags.DEFINE_bool(
    "embed_node_attr", default=False, help="Embed node attributes.")
flags.DEFINE_integer(
    "dim_node_embed", default=-1,
    help="Number of dimensions in node embeddings.")
flags.DEFINE_bool(
    "embed_edge_attr", default=False, help="Embed edge attributes.")
flags.DEFINE_integer(
    "dim_edge_embed", default=-1,
    help="Number of dimensions in edge embeddings.")
flags.DEFINE_bool(
    "use_self_loop", default=False, help="Add/Remove self-loop edges.")
flags.DEFINE_string(
    "gnn_impl", default=GNN_SPARSE,
    help="GNN implementation (sparse|dense|disable).")
flags.DEFINE_string(
    "gnn_attention", default=None,
    help="Function to use for normalizing attention weights.")
flags.DEFINE_integer(
    "gnn_num_layers", default=2,
    help="Number of recurrent layers in multilayer GNN.")
flags.DEFINE_integer(
    "gnn_num_heads", default=4,
    help="Number of attention heads to use in GNN.")
flags.DEFINE_integer(
    "gnn_dim_key", default=16,
    help="Number of dimensions in the keys of GNN attention.")
flags.DEFINE_integer(
    "gnn_dim_value", default=16,
    help="Number of dimensions in the values of GNN attention.")
flags.DEFINE_bool(
    "gnn_layer_norm_in", default=True,
    help="If true, apply layer normalization to the input of GNN.")
flags.DEFINE_bool(
    "gnn_layer_norm_out", default=False,
    help="If true, apply layer normalization to the output of GNN.")
flags.DEFINE_bool(
    "gnn_recurrent", default=False,
    help="If true, share the GNN parameters in multilayer GNNs.")
flags.DEFINE_string(
    "gnn_combiner", default=COMBINER_LSTM,
    help="Function to use for aggregating multilayer updates (lstm|gru|add).")
flags.DEFINE_string(
    "gnn_messenger", default=MESSENGER_BINARY,
    help="Messenger to use for computing messages (unary|binary).")
flags.DEFINE_string(
    "gnn_activation", default="linear",
    help="Activation function for GNN outputs.")

flags.DEFINE_bool(
    "learn_init_states", default=True,
    help="Use trainable initial belief/latent states.")
flags.DEFINE_integer(
    "trans_gnn_num_layers", default=1,
    help="Number of GNN layers in generative model.")
flags.DEFINE_integer(
    "init_mix_num_components", default=1,
    help="Number of mixture components in the prior distribution.")
flags.DEFINE_integer(
    "trans_mix_num_components", default=1,
    help="Number of mixture components in the trans distribution.")
flags.DEFINE_integer(
    "trans_mlp_num_layers", default=0,
    help="Number of hidden layers in transition MLP.")
flags.DEFINE_string(
    "trans_activation", default="tanh",
    help="Type of activation to use for transition MLP (loc).")
flags.DEFINE_bool(
    "trans_layer_norm", default=True,
    help="If true, apply layer normalization to loc.")
flags.DEFINE_string(
    "trans_scale_activation", default="softplus",
    help="Type of activation to use for transition MLP (scale).")
flags.DEFINE_float(
    "trans_scale_shift", default=1e-3,
    help="Positive shift to add on the stddev of transition prior.")
flags.DEFINE_bool(
    "trans_scale_identical", default=False,
    help="If true, use identical variance.")
flags.DEFINE_bool(
    "trans_skip_conn", default=True, help="z_t = z_{t-1} + delta.")
flags.DEFINE_bool(
    "trans_ar", default=True, help="Add autoregressive connection.")
flags.DEFINE_integer(
    "trans_global_low_rank", default=0,
    help="Number of perturb factors in global MVN covariance.")
flags.DEFINE_integer(
    "trans_local_low_rank", default=0,
    help="Number of perturb factors in local MVN covariance.")
flags.DEFINE_integer(
    "rnn_num_layers", default=1, help="Number of RNN layers.")

flags.DEFINE_integer(
    "trans_flow_num_layers", default=0,
    help="Number of normalizing flow layers.")
flags.DEFINE_integer(
    "proposal_flow_num_layers", default=0,
    help="Number of normalizing flow layers.")
flags.DEFINE_bool(
    "reuse_gen_flow", default=False,
    help="Reuse the normalizing flows of the generative model.")
flags.DEFINE_string(
    "flow_mv_factor", default="qr",
    help="Trainable 1x1 convolution implementation, (qr|lu).")
flags.DEFINE_bool(
    "flow_skip_conn", default=False,
    help="Add skip connection to the base distribution.")
flags.DEFINE_bool(
    "global_flow", default=False, help="Enable flow for global states.")

flags.DEFINE_integer(
    "emit_low_rank", default=0,
    help="Number of perturb factors in emit MVN covariance.")
flags.DEFINE_integer(
    "emit_mix_num_components", default=1,
    help="Number of mixture components in the emit distribution.")
flags.DEFINE_integer(
    "emit_mlp_num_layers", default=1,
    help="Number of hidden layers in emission MLP.")
flags.DEFINE_string(
    "emit_activation", default="linear",
    help="Type of activation to use for emission MLP (loc).")
flags.DEFINE_string(
    "emit_scale_activation", default="softplus",
    help="Type of activation to use for emission MLP (scale).")
flags.DEFINE_float(
    "emit_scale_shift", default=1e-3,
    help="Positive shift to add on the stddev of emission likelihood.")
flags.DEFINE_bool(
    "emit_scale_identical", default=True,
    help="If true, use identical variance.")
flags.DEFINE_bool(
    "emit_neg_binomial", default=False,
    help="If true, use a negative binomial likelihood.")
flags.DEFINE_string(
    "emit_loc_scale_type", default="normal",
    help="(laplace|logistic|gumbel), use a non-Normal likelihood instead.")
flags.DEFINE_bool(
    "emit_non_markov", default=True,
    help="If true, let x_t condition on h_t.")
flags.DEFINE_bool(
    "emit_identity", default=False,
    help="If true, output identity mapping of the leading dims" +
    "of latent states as observations.")

flags.DEFINE_integer(
    "rand_walk_len", default=100,
    help="Length of random walk to perform in each batch.")
flags.DEFINE_integer(
    "min_num_nodes", default=32,
    help="Minimum number of nodes returned by graph sampler.")
flags.DEFINE_float(
    "rand_walk_restart_prob", default=0.25,
    help="Probability for jumping back to initial vertex.")
flags.DEFINE_integer(
    "rand_walk_len_periods", default=1,
    help="Number of periods for scheduling the random walk length")
flags.DEFINE_float(
    "rand_walk_len_tempering", default=0.0,
    help="=0: max; >0: min->max->min; <0: max->min->max")
flags.DEFINE_bool(
    "rand_walk_rand_len", default=False,
    help="If true, randomly walk up to random number of steps")
flags.DEFINE_integer(
    "skip_gram_win_size", default=2,
    help="Window size to use in skipgram sampling")
flags.DEFINE_integer(
    "num_neg_samples", default=1,
    help="Number of vertices to pick for each vertex in negative sampling.")

flags.DEFINE_string(
    "dataset", default=os.path.join("datasets", "test"),
    help="Directory where data is stored.")
flags.DEFINE_string(
    "dataset_type", default=SAME_GRAPH,
    help="Dataset type, (same|diff).")
flags.DEFINE_string(
    "eval_dataset", default=None,
    help="Specify the dataset to evaluate.")
flags.DEFINE_string(
    "logdir", default=os.path.join("logs", "model"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "save_steps", default=25,
    help="Frequency at which to save checkpoints.")
flags.DEFINE_integer(
    "eval_steps", default=10,
    help="Frequency at which to evaluate the model.")
flags.DEFINE_integer(
    "pred_steps", default=10,
    help="Frequency at which to test prediction accuracy.")
flags.DEFINE_bool(
    "fake_data", default=False,
    help="If true, uses fake data instead.")
flags.DEFINE_bool(
    "delete_existing", default=False,
    help="If true, deletes existing `model_dir` directory.")
flags.DEFINE_bool(
    "zscore", default=False,
    help="If true, normalize the dataset into Z-score.")
flags.DEFINE_list(
    "preprocess_shift", default=None, help="(dataset - shift) / scale.")
flags.DEFINE_list(
    "preprocess_scale", default=None, help="(dataset - shift) / scale.")
flags.DEFINE_integer(
    "win_size", default=None, help="Number of time steps per window.")
flags.DEFINE_integer(
    "stride", default=None, help="Number of time steps between two windows.")
flags.DEFINE_bool(
    "train_on_rand_win", default=False,
    help="If true, use randomly sliced windows for training.")
flags.DEFINE_integer(
    "rand_win_size_lo", default=32,
    help="Lower bound of random window size.")
flags.DEFINE_integer(
    "rand_win_size_hi", default=128,
    help="Upper bound of random window size.")
flags.DEFINE_bool(
    "eval_only", default=False,
    help="Disable training and perform evaluation only.")
flags.DEFINE_bool(
    "sample_only", default=False,
    help="Sample future obseravtions only; don't quantity prediction error.")
flags.DEFINE_integer(
    "sample_burn_in_steps", default=0,
    help="How many steps to use for burn in.")
flags.DEFINE_string(
    "save_pred_to_file", default=None,
    help="Save predictions to given file.")
flags.DEFINE_string(
    "save_pred_prefix", default=None,
    help="(partial|full), add prefix when saving predictions to given file.")
flags.DEFINE_bool(
    "pred_every_step", default=True,
    help="Predict at every time step.")
flags.DEFINE_bool(
    "pred_resample_init", default=True,
    help="Whether or not to resample the initial particles before prediction.")
flags.DEFINE_bool(
    "pred_ar_filtering", default=False,
    help="Always use one-step prediction and feed prediction into proposal.")


FLAGS = flags.FLAGS


def print_metrics(metrics):
    lines = "\n"
    for key in metrics.keys():
        value = metrics.get(key)
        lines += "{}:\n{!s}\n".format(key, value)
    tf.logging.info(lines)


def make_step_fn(dataloaders, dataset_transform, summary_writers,
                 params, config, options=None, run_metadata=None):
    train_dataloader, eval_dataloader = dataloaders
    train_writer, eval_writer = summary_writers

    dim_observ = params["dim_observs"]
    dim_node_attr = params["dim_node_attrs"]
    dim_edge_attr = params["dim_edge_attrs"]
    dim_time_attr = params["dim_time_attrs"]
    dim_hidden, dim_latent = config.dim_hidden, config.dim_latent

    rnn_num_layers = params["rnn_num_layers"]
    dim_belief_state = 2 * rnn_num_layers * dim_hidden
    dim_history = 2 * rnn_num_layers * dim_hidden

    reset = tf.placeholder(tf.bool, name="reset")

    # (B, N, dH)
    belief_states_shape = [None, None, dim_belief_state]
    belief_states = tf.placeholder(
        tf.float32, shape=belief_states_shape, name="belief_states"
    )
    # (B, S, dh)
    global_latent_histories = tf.placeholder(
        tf.float32, shape=[None, None, dim_history],
        name="global_latent_histories"
    )
    # (B, S, N, dh)
    local_latent_histories = tf.placeholder(
        tf.float32, shape=[None, None, None, dim_history],
        name="local_latent_histories"
    )
    # (B, S, dz)
    global_latent_states = tf.placeholder(
        tf.float32, shape=[None, None, dim_latent],
        name="global_latent_states"
    )
    # (B, S, N, dz)
    local_latent_states = tf.placeholder(
        tf.float32, shape=[None, None, None, dim_latent],
        name="local_latent_states"
    )
    latent_histories = (global_latent_histories, local_latent_histories)
    latent_states = (global_latent_states, local_latent_states)
    STATES = PersistentStates(
        reset=reset,
        belief_states=belief_states,
        latent_histories=latent_histories,
        latent_states=latent_states
    )
    states_key = (reset, belief_states, latent_histories, latent_states)

    # (T, B, N, dx)
    observations = tf.placeholder(
        tf.float32, shape=[None, None, None, dim_observ],
        name="observations"
    )
    # (T, H, B, N, dx)
    labels = tf.placeholder(
        tf.float32, shape=[None, None, None, None, dim_observ],
        name="labels"
    )

    # (B, E, 2)
    edges = tf.placeholder(
        tf.int32, shape=[None, None, 2], name="edges"
    )
    # (B, N)
    node_mask = tf.placeholder(
        tf.int32, shape=[None, None], name="node_mask"
    )
    # (B, E)
    edge_mask = tf.placeholder(
        tf.int32, shape=[None, None], name="edge_mask"
    )
    # (B, N)
    center_mask = tf.placeholder(
        tf.int32, shape=[None, None], name="center_mask"
    )

    # (B, N/E/T, d*)
    node_attrs = edge_attrs = time_attrs = None
    if dim_node_attr > 0:
        node_attrs = tf.placeholder(
            tf.float32, shape=[None, None, dim_node_attr]
        )
        tf.logging.info("Number of node attributes: %d" % dim_node_attr)
    if dim_edge_attr > 0:
        edge_attrs = tf.placeholder(
            tf.float32, shape=[None, None, dim_edge_attr]
        )
        tf.logging.info("Number of edge attributes: %d" % dim_edge_attr)
    if dim_time_attr > 0:
        time_attrs = tf.placeholder(
            tf.float32, shape=[None, None, dim_time_attr]
        )
        tf.logging.info("Number of time attributes: %d" % dim_time_attr)

    mode = tf.placeholder(tf.string, name="mode")

    SOURCES = FlowSources(
        observations=observations,
        edges=edges,
        center_mask=center_mask,
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        time_attrs=time_attrs,
        labels=labels,
        mode=mode
    )

    MODEL = build_shared_model(params, config)
    SINKS = build_tf_graph(
        MODEL, STATES, SOURCES, dataset_transform, params, config
    )

    global_step = tf.train.get_or_create_global_step()

    train_summary_all = tf.summary.merge([
        tf.summary.merge_all(),
        tf.summary.merge_all(key=SUMMARY_TRAIN)
    ])
    eval_summary_all = tf.summary.merge([
        tf.summary.merge_all(),
        tf.summary.merge_all(key=SUMMARY_EVAL),
    ])
    quick_predict_summary_all = tf.summary.merge_all(
        key=SUMMARY_QUICK_PREDICT
    )
    slow_predict_summary_all = tf.summary.merge_all(
        key=SUMMARY_SLOW_PREDICT
    )

    def states_zero_value(batch, num_samples):
        batch_size = np.shape(batch.observations)[-3]
        num_vertices = np.shape(batch.observations)[-2]
        return (
            True,
            np.zeros([batch_size, num_vertices, dim_belief_state]),
            (
                np.zeros([num_samples, batch_size, dim_history]),
                np.zeros([num_samples, batch_size, num_vertices, dim_history])
            ), (
                np.zeros([num_samples, batch_size, dim_latent]),
                np.zeros([num_samples, batch_size, num_vertices, dim_latent])
            ),
        )

    def gen_feed_dict(batch, num_pred_steps):
        extra_features = {}
        if dim_node_attr > 0:
            extra_features[SOURCES.node_attrs] = batch.node_attrs
        if SOURCES.edge_attrs is not None:
            extra_features[SOURCES.edge_attrs] = batch.edge_attrs
        if SOURCES.time_attrs is not None:
            extra_features[SOURCES.time_attrs] = batch.time_attrs
        feed_dict = {
            SOURCES.edges: batch.edges,
            SOURCES.center_mask: batch.center_mask,
            SOURCES.node_mask: batch.node_mask,
            SOURCES.edge_mask: batch.edge_mask,
            **extra_features
        }
        X = batch.observations
        win = num_pred_steps
        if win > 0:
            feed_dict[observations] = X[:-win]
            feed_dict[labels] = sliding_window_view(X, win)[1:]
        else:
            feed_dict[observations] = X
            # We ensure that predict_ops will be not evaluated if win <= 0,
            # so feeding arbitrary data into `labels` is OK.
            feed_dict[labels] = sliding_window_view(X, 1)
        return feed_dict

    def evaluate(session, batch):
        step, summaries, metrics = session.run(
            [global_step, eval_summary_all, SINKS.eval_metrics],
            feed_dict={
                **gen_feed_dict(batch, 0),
                states_key: states_zero_value(batch, config.eval_num_samples),
                mode: MODE_EVAL
            }
        )
        eval_writer.add_summary(summaries, step)
        tf.logging.info("Evaluation result at step {}:".format(step))
        print_metrics(metrics)
        return metrics

    def predict(session, batch, pred_mode, num_samples, num_pred_steps,
                predict_op, predict_metrics, summary_ops, writer):
        step, predictions, metrics, summaries = session.run(
            [global_step, predict_op, predict_metrics, summary_ops],
            feed_dict={
                **gen_feed_dict(batch, num_pred_steps),
                states_key: states_zero_value(batch, num_samples),
                mode: pred_mode
            }
        )
        writer.add_summary(summaries, step)
        tf.logging.info(
            "{}: prediction errors at step {}:".format(pred_mode, step)
        )
        print_metrics(metrics)
        return predictions, metrics

    def sample(session, batch, num_samples, burn_in_steps, predict_op):
        step, predictions = session.run(
            [global_step, predict_op],
            feed_dict={
                **gen_feed_dict(batch, batch.num_time_steps - burn_in_steps),
                states_key: states_zero_value(batch, num_samples),
                mode: MODE_SAMPLE
            }
        )
        return predictions

    def logging_overall_prediction_metrics(writer, step, dicts):
        merged_metrics = {
            k: [d.get(k) for d in dicts]
            for k in dicts[0].keys()
        }
        reduced_metrics = dict()
        for pair in itertools.product(
            ["MAE", "MSE", "MAPE"],
            ["AVG", "MEDIAN"]
        ):
            key = "{}_{}".format(pair[1], pair[0])
            reduced_metrics[key] = np.mean(merged_metrics.get(key), axis=0)
        tf.logging.info(
            "Overall prediction error at step {}:\n".format(step) +
            "\n".join(
                ["{}:\n{!s}".format(k, v) for k, v in reduced_metrics.items()]
            )
        )
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="ALL_" + k, simple_value=np.mean(v))
            for k, v in reduced_metrics.items()
        ])
        writer.add_summary(summary, step)

    def evaluate_all(session):
        step = session.run(global_step)
        dicts, arrays = [], []

        for i, batch in enumerate(eval_dataloader, 0):
            if i == 0:
                tf.logging.info("EVAL: batch.observations.shape = {}".format(
                    batch.observations.shape
                ))

            evaluate(session, batch)

            if config.eval_num_pred_steps <= 0:
                continue

            if config.sample_only:
                predictions = sample(
                    session=session, batch=batch,
                    num_samples=config.eval_num_pred_samples,
                    burn_in_steps=config.sample_burn_in_steps,
                    predict_op=SINKS.slow_predictions
                )
            else:
                predictions, predict_metrics = predict(
                    session=session, batch=batch,
                    pred_mode=MODE_SLOW_PREDICT,
                    num_samples=config.eval_num_pred_samples,
                    num_pred_steps=config.eval_num_pred_steps,
                    predict_op=SINKS.slow_predictions,
                    predict_metrics=SINKS.slow_predict_metrics,
                    summary_ops=slow_predict_summary_all,
                    writer=eval_writer
                )
                dicts.append(predict_metrics)

            if config.save_pred_to_file is not None:
                # (S, T + H, B, N, dx)
                if config.save_pred_prefix is not None:
                    S = predictions.shape[0]
                    prefix = np.broadcast_to(
                        batch.observations,
                        [S, *batch.observations.shape]
                    )
                    if config.sample_only and \
                            config.save_pred_prefix == "partial":
                        B = config.sample_burn_in_steps
                        prefix = prefix[:, :B, ...]
                    predictions = np.concatenate(
                        [prefix, predictions], axis=1
                    )
                arrays.append(predictions)

        if config.eval_num_pred_steps <= 0:
            return

        if config.save_pred_to_file is not None:
            np.savez_compressed(config.save_pred_to_file, *arrays)
            tf.logging.info("Predictions have been saved to {} .".format(
                config.save_pred_to_file
            ))

        if not config.sample_only:
            logging_overall_prediction_metrics(eval_writer, step, dicts)

    def train(session, saver, batch, tbptt_states):
        feed_dict = gen_feed_dict(batch, 0)

        summaries, step, _, loss, new_states = session.run(
            [
                train_summary_all, global_step,
                SINKS.train_op, SINKS.train_loss, SINKS.new_states
            ],
            feed_dict={
                **feed_dict,
                states_key: tbptt_states,
                mode: MODE_TRAIN,
            },
            options=options,
            run_metadata=run_metadata
        )
        tbptt_states = tuple([False, *new_states])

        if step % config.save_steps == 0:
            train_writer.add_summary(summaries, step)

        if step % config.pred_steps == 0 and config.train_num_pred_steps > 0:
            predict(
                session=session, batch=batch,
                pred_mode=MODE_QUICK_PREDICT,
                num_samples=config.train_num_pred_samples,
                num_pred_steps=config.train_num_pred_steps,
                predict_op=SINKS.quick_predictions,
                predict_metrics=SINKS.quick_predict_metrics,
                summary_ops=quick_predict_summary_all,
                writer=train_writer
            )

        if step % config.save_steps == 0:
            path = saver.save(
                sess=session,
                save_path=os.path.join(config.logdir, "model"),
                global_step=step
            )
            tf.logging.info(
                "Latest checkpoint has been saved to {}".format(path)
            )

        return tbptt_states

    def train_all(session, saver):
        tbptt_states = None

        for i, batch in enumerate(train_dataloader, 0):
            if i == 0:
                tf.logging.info("TRAIN: batch.observations.shape = {}".format(
                    batch.observations.shape
                ))
            zero_states = states_zero_value(batch, config.num_samples)
            if (not config.tbptt) or (tbptt_states is None):
                tbptt_states = zero_states

            tbptt_states = train(session, saver, batch, tbptt_states)

    num_epoches = -1

    def step_fn(session, saver):
        nonlocal num_epoches

        if config.eval_only:
            evaluate_all(session)
            return True

        num_epoches += 1
        tf.logging.info("New Epoch {} ...".format(num_epoches))

        train_all(session, saver)

        step = session.run(global_step)
        if step > config.num_steps:
            return True

        if config.disable_eval or step == 0 or step % config.eval_steps != 0:
            return False

        evaluate_all(session)
        return False

    return step_fn


def count_num_trainable_params():
    return np.sum([
        np.prod(v.get_shape().as_list())
        for v in tf.trainable_variables()
    ])


def save_trace(session, run_metadata, config):
    global_step = tf.train.get_or_create_global_step()
    step = session.run(global_step)
    tl = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = tl.generate_chrome_trace_format()
    filename = os.path.join(
        config.logdir, "timeline_step_{}.json".format(step))
    with open(filename, "w") as f:
        f.write(chrome_trace)


def recover_from_latest_checkpoint(session, saver, config):
    latest_checkpoint = tf.train.latest_checkpoint(config.logdir)
    if latest_checkpoint is not None:
        tf.logging.info("Recover states from latest checkpoint {} ...".format(
            latest_checkpoint
        ))
        saver.restore(session, latest_checkpoint)
    else:
        tf.logging.info("No checkpoint was found in {}.".format(config.logdir))


def run(params, config):
    train_dataloader, eval_dataloader, \
        train_dataset = build_dataloader(config)

    dataset_transform = train_dataset.transform
    params["dim_observs"] = train_dataset.dim_observs
    params["dim_node_attrs"] = train_dataset.dim_node_attrs
    params["dim_edge_attrs"] = train_dataset.dim_edge_attrs
    params["dim_time_attrs"] = train_dataset.dim_time_attrs

    summary_dir = os.path.join(config.logdir, "summaries")
    train_writer = SummaryWriterCache.get(
        os.path.join(summary_dir, "train"))
    eval_writer = tf.summary.FileWriter(
        os.path.join(summary_dir, "eval"))

    G = tf.Graph()
    with G.as_default():
        tf.random.set_random_seed(config.seed)

        options, run_metadata = None, None
        if config.trace:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        sess = tf.Session()
        tf.keras.backend.set_session(sess)

        step_fn = make_step_fn(
            (train_dataloader, eval_dataloader),
            dataset_transform,
            (train_writer, eval_writer),
            params, config,
            options=options, run_metadata=run_metadata
        )

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)
        recover_from_latest_checkpoint(sess, saver, config)

        G.finalize()
        tf.logging.info("Graph has been finalized.")
        tf.logging.info("Number of trainable parameters = {}.".format(
            count_num_trainable_params()
        ))

        with sess.as_default():
            should_stop = False
            while not should_stop:
                should_stop = step_fn(sess, saver)
                if config.trace:
                    save_trace(sess, run_metadata, config)
        sess.close()
        tf.logging.info("DONE.")


def main(argv):
    del argv

    params = FLAGS.flag_values_dict()
    params_as_json = json.dumps(params, indent=4)
    tf.logging.info("Command-line flags:\n{}".format(params_as_json))

    assert not (FLAGS.eval_only and FLAGS.delete_existing)
    assert not (FLAGS.save_pred_to_file is not None and not FLAGS.eval_only)

    dir = FLAGS.logdir
    if FLAGS.delete_existing and tf.gfile.Exists(dir):
        tf.logging.warning("Overwriting old directory at {}".format(dir))
        tf.gfile.DeleteRecursively(dir)
        tf.gfile.MakeDirs(dir)
    elif tf.gfile.Exists(dir):
        tf.logging.info("Restoring states from {}".format(dir))
    else:
        tf.logging.info("Creating new directory at {}".format(dir))
        tf.gfile.MakeDirs(dir)

    run(params, FLAGS)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
