from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import tensorboard as tb
from tensorflow import keras

from generative import NonMarkovModel, ExternalSequences
from auxiliary import ZForcing, EdgeClassifier, CPC, DGI, MaskedGNNDecoder, MIX

from dmm import MarkovModel
from vi import VI
from vsmc import VSMC
from proposal import Proposal, IndepProposal
from proposal import SUMMARIZER_LSTM, SUMMARIZER_RGNN
from proposal import FactorizedWrapper
from hvae import LearnableHIS
from sched import InterleavingScheduler

from tgraph import RuntimeGraph
from gnn import GNNConfig, GNN_DENSE
from gnn import ATTENTION_SOFTMAX, ATTENTION_UNIFORM
from gnn import READOUT_MEAN_MAX

import metrics
import util

import functools
import itertools

tfd = tfp.distributions

MODE_TRAIN = "TRAIN"
MODE_EVAL = "EVAL"
MODE_QUICK_PREDICT = "QUICK-PREDICT"
MODE_SLOW_PREDICT = "SLOW-PREDICT"
MODE_SAMPLE = "SAMPLE"

SUMMARY_TRAIN = "SUMMARY_TRAIN"
SUMMARY_EVAL = "SUMMARY_EVAL"
SUMMARY_QUICK_PREDICT = "SUMMARY_QUICK_PREDICT"
SUMMARY_SLOW_PREDICT = "SUMMARY_SLOW_PREDICT"

AUX_ADJ = "adj"
AUX_CPC = "cpc"
AUX_DGI = "dgi"
AUX_ZF = "zf"
AUX_MASK = "mask"
AUX_MIX = "mix"


class Model(object):
    def __init__(self, gen_model, proposal_model, aux_model,
                 hamiltonian_is=None, interleaving_scheduler=None,
                 aux_weight=1.0):
        self.gen_model = gen_model
        self.proposal_model = proposal_model
        self.aux_model = aux_model
        self.hamiltonian_is = hamiltonian_is
        self.interleaving_scheduler = interleaving_scheduler
        self.aux_weight = aux_weight


class FlowSources(object):
    def __init__(self, observations, edges,
                 center_mask, node_mask, edge_mask,
                 labels, mode,
                 node_attrs=None, edge_attrs=None, time_attrs=None):
        self.observations = observations
        self.edges = edges
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs
        self.time_attrs = time_attrs
        self.center_mask = center_mask
        self.node_mask = node_mask
        self.edge_mask = edge_mask
        self.labels = labels
        self.mode = mode


class FlowSinks(object):
    def __init__(self, train_op, new_states, train_loss, eval_metrics,
                 quick_predictions, quick_predict_metrics,
                 slow_predictions, slow_predict_metrics):
        self.train_op = train_op
        self.new_states = new_states
        self.train_loss = train_loss
        self.eval_metrics = eval_metrics
        self.quick_predictions = quick_predictions
        self.quick_predict_metrics = quick_predict_metrics
        self.slow_predictions = slow_predictions
        self.slow_predict_metrics = slow_predict_metrics


class PersistentStates(object):
    def __init__(self, reset, belief_states, latent_histories, latent_states):
        self.reset = reset
        self.belief_states = belief_states
        self.latent_histories = latent_histories
        self.latent_states = latent_states

    def trainable(self):
        initializer = tf.initializers.glorot_uniform

        belief_state = tf.get_variable(
            "initial_belief_state", [util.dim(self.belief_states)],
            trainable=True, initializer=initializer()
        )
        global_latent_history = tf.get_variable(
            "initial_global_history", [util.dim(self.latent_histories[0])],
            trainable=True, initializer=initializer()
        )
        local_latent_history = tf.get_variable(
            "initial_local_history", [util.dim(self.latent_histories[1])],
            trainable=True, initializer=initializer()
        )
        global_latent_state = tf.get_variable(
            "initial_global_state", [util.dim(self.latent_states[0])],
            trainable=True, initializer=initializer()
        )
        local_latent_state = tf.get_variable(
            "initial_local_state", [util.dim(self.latent_states[1])],
            trainable=True, initializer=initializer()
        )

        z = tf.zeros_like

        belief_states = tf.math.add(z(self.belief_states), belief_state)
        latent_histories = (
            tf.math.add(z(self.latent_histories[0]), global_latent_history),
            tf.math.add(z(self.latent_histories[1]), local_latent_history),
        )
        latent_states = (
            tf.math.add(z(self.latent_states[0]), global_latent_state),
            tf.math.add(z(self.latent_states[1]), local_latent_state),
        )

        original_states = (
            self.belief_states, self.latent_histories, self.latent_states
        )
        states = util.select_nested(
            self.reset,
            (belief_states, latent_histories, latent_states),
            original_states
        )
        states = util.nested_set_shape_like(states, original_states)
        return PersistentStates(self.reset, *states)


class Template(object):
    def __init__(self, estimator, predictor):
        self._estimator = estimator
        self._predictor = predictor

    def estimate(self, *args, **kwargs):
        return self._estimator(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._predictor(*args, **kwargs)


def build_shared_model(params, config):
    if config.train_obj == "vi" and config.analytic_kl and \
            config.trans_mix_num_components != 1:
        raise ValueError("Analytic KL cannot be applied to mixtures.")

    gnn_attention = config.gnn_attention
    if gnn_attention is None:
        gnn_attention = ATTENTION_UNIFORM if config.mini_batch \
            else ATTENTION_SOFTMAX
    elif gnn_attention == ATTENTION_SOFTMAX and config.mini_batch:
        raise ValueError("Bad configuration.")

    assert not (config.embed_node_attr and params["dim_node_attrs"] <= 0)
    assert not (config.embed_edge_attr and params["dim_edge_attrs"] <= 0)
    assert not (config.learn_node_embed and config.const_num_nodes is None)
    dim_node_attr = (
        params["dim_node_embed"]
        if config.embed_node_attr else params["dim_node_attrs"]
    ) + (
        params["dim_node_embed"]
        if config.learn_node_embed else 0
    )
    dim_edge_attr = params["dim_edge_embed"] \
        if config.embed_edge_attr else params["dim_edge_attrs"]

    gnn_config = GNNConfig(
        num_heads=params["gnn_num_heads"],
        dim_input=None,
        dim_key=params["gnn_dim_key"],
        dim_value=params["gnn_dim_value"],
        dim_node_attr=dim_node_attr,
        dim_edge_attr=dim_edge_attr,
        impl=config.gnn_impl,
        attention=gnn_attention,
        messenger=config.gnn_messenger,
        activation=config.gnn_activation,
        layer_norm_in=params["gnn_layer_norm_in"],
        layer_norm_out=params["gnn_layer_norm_out"],
        skip_conn=False,
        num_layers=1,
        combiner=params["gnn_combiner"],
        recurrent=params["gnn_recurrent"],
        rnn_num_layers=params["rnn_num_layers"],
        readout=READOUT_MEAN_MAX,
        parallel_iterations=params["parallel_iterations"],
        swap_memory=params["swap_memory"]
    )

    trans_gnn_config = gnn_config.clone()
    trans_gnn_config.num_layers = params["trans_gnn_num_layers"]

    gen_model_params = dict(
        dim_hidden=params["dim_hidden"],
        dim_observ=params["dim_observs"],
        dim_latent=params["dim_latent"],
        dim_mlp=params["dim_mlp"],
        dim_global_input=params["dim_time_attrs"],
        const_num_nodes=params["const_num_nodes"],
        gnn_config=trans_gnn_config.clone(),
        rnn_num_layers=params["rnn_num_layers"],
        init_mix_num_components=params["init_mix_num_components"],
        trans_mix_num_components=params["trans_mix_num_components"],
        trans_mlp_num_layers=params["trans_mlp_num_layers"],
        trans_activation=params["trans_activation"],
        trans_layer_norm=params["trans_layer_norm"],
        trans_scale_activation=params["trans_scale_activation"],
        trans_scale_shift=params["trans_scale_shift"],
        trans_scale_identical=params["trans_scale_identical"],
        trans_skip_conn=params["trans_skip_conn"],
        trans_ar=params["trans_ar"],
        trans_global_low_rank=params["trans_global_low_rank"],
        trans_local_low_rank=params["trans_local_low_rank"],
        trans_global_flow=config.global_flow,
        trans_flow_num_layers=params["trans_flow_num_layers"],
        trans_flow_mv_factor=params["flow_mv_factor"],
        trans_flow_skip_conn=config.flow_skip_conn,
        emit_low_rank=params["emit_low_rank"],
        emit_mix_num_components=params["emit_mix_num_components"],
        emit_mlp_num_layers=params["emit_mlp_num_layers"],
        emit_activation=params["emit_activation"],
        emit_scale_activation=params["emit_scale_activation"],
        emit_scale_shift=params["emit_scale_shift"],
        emit_scale_identical=params["emit_scale_identical"],
        emit_neg_binomial=params["emit_neg_binomial"],
        emit_loc_scale_type=params["emit_loc_scale_type"],
        emit_non_markov=params["emit_non_markov"],
        emit_identity=params["emit_identity"]
    )
    if config.markov:
        gen_model = MarkovModel(**gen_model_params)
    else:
        gen_model = NonMarkovModel(**gen_model_params)

    perturb_noise_scale = tf.train.linear_cosine_decay(
        config.noise_scale,
        tf.train.get_or_create_global_step(),
        config.noise_scale_decay_steps or config.num_steps,
        alpha=0.0, beta=config.noise_scale_min_ratio
    )
    tb.summary.scalar("perturb_noise_scale", perturb_noise_scale)

    proposal_gnn_config = gnn_config.clone()
    proposal_gnn_config.num_layers = params["proposal_gnn_num_layers"]

    proposal_params = dict(
        model=gen_model,
        dim_mlp=params["dim_mlp"],
        mlp_num_layers=params["trans_mlp_num_layers"],
        rnn_num_layers=params["rnn_num_layers"],
        global_flow=config.global_flow,
        flow_num_layers=params["proposal_flow_num_layers"],
        flow_mv_factor=params["flow_mv_factor"],
        flow_skip_conn=config.flow_skip_conn,
        global_low_rank=params["trans_global_low_rank"],
        local_low_rank=params["trans_local_low_rank"],
        loc_activation=params["proposal_loc_activation"],
        loc_layer_norm=params["proposal_loc_layer_norm"],
        scale_activation=params["trans_scale_activation"],
        scale_shift=params["trans_scale_shift"],
        scale_identical=config.proposal_scale_identical,
        gnn_config=proposal_gnn_config.clone(),
        use_belief=config.use_belief,
        use_lookahead=config.use_lookahead,
        use_skip_conn=config.use_skip_conn,
        use_gated_adder=config.use_gated_adder,
        reuse_gen_flow=config.reuse_gen_flow,
        denoising=config.denoising,
        noise_scale=perturb_noise_scale
    )

    if config.proposal == "indep":
        proposal_model = IndepProposal(
            **proposal_params, summarize_unit=SUMMARIZER_LSTM
        )
    elif config.proposal == "joint" and not config.mini_batch:
        proposal_model = Proposal(
            **proposal_params, summarize_unit=SUMMARIZER_RGNN
        )
    else:
        raise ValueError("Unknown/Invalid proposal type.")

    proposal_model = FactorizedWrapper(proposal_model)

    interleaving_rate = tf.train.polynomial_decay(
        config.interleaving_rate,
        tf.train.get_or_create_global_step(),
        config.interleaving_decay_steps or (config.num_steps // 2),
        (config.interleaving_rate_min_ratio * config.interleaving_rate),
        power=1.0
    )
    tb.summary.scalar("interleaving_rate", interleaving_rate)

    interleaving_scheduler = InterleavingScheduler(
        prefix_length=(config.prefix_length if config.prefixing else 0),
        rate=interleaving_rate,
        randomly=config.interleaving_randomly,
        refresh_last_step=False
    )
    if not config.interleaving:
        assert not config.prefixing
        interleaving_scheduler = None

    hamiltonian_is = None
    if config.his:
        hamiltonian_is = LearnableHIS(
            global_num_dims=params["dim_latent"],
            local_num_dims=params["dim_latent"],
            num_steps=params["his_num_leapfrog_steps"],
            max_step_size=params["his_max_step_size"],
            mass_scale=params["his_mass_scale"]
        )

    if config.aux_task is not None and not config.use_lookahead:
        tf.logging.warning(
            "Auxiliary loss is optimized without using lookahead information."
        )
    aux_params = dict(
        dim_latent=params["dim_latent"],
        dim_summary=params["dim_hidden"],
    )
    if config.aux_task == AUX_ADJ:
        aux_model = EdgeClassifier(**aux_params)
    elif config.aux_task == AUX_CPC:
        aux_model = CPC(**aux_params)
    elif config.aux_task == AUX_DGI:
        aux_model = DGI(**aux_params)
    elif config.aux_task == AUX_ZF:
        aux_model = ZForcing(
            **aux_params,
            dim_mlp=params["dim_mlp"], mlp_num_layers=1,
            num_future_steps=params["aux_zf_num_steps"]
        )
    elif config.aux_task == AUX_MASK:
        aux_model = MaskedGNNDecoder(
            **aux_params,
            gnn_config=gnn_config.clone(),
            dim_mlp=params["dim_mlp"],
            num_masked_nodes=params["aux_mask_num_nodes"],
            all_at_once=params["aux_mask_all_at_once"]
        )
    elif config.aux_task == AUX_MIX:
        aux_model = MIX(
            **aux_params,
            dim_observ=params["dim_observs"],
            gnn_config=gnn_config.clone(),
            dim_mlp=params["dim_mlp"], mlp_num_layers=2,
            cpc_scale=params["aux_cpc_scale"],
            cpc_state=params["aux_cpc_state"],
            dgi_scale=params["aux_dgi_scale"],
            zf_scale=params["aux_zf_scale"],
            zf_num_future_steps=params["aux_zf_num_steps"],
            mask_scale=params["aux_mask_scale"],
            mask_num_nodes=params["aux_mask_num_nodes"],
            mask_all_at_once=params["aux_mask_all_at_once"]
        )
    elif config.aux_task is not None:
        raise ValueError("Unknown auxiliary task: " + config.aux_task)
    else:
        aux_model = None

    aux_weight = tf.train.polynomial_decay(
        config.aux_weight,
        tf.train.get_or_create_global_step(),
        config.aux_weight_decay_steps or config.num_steps,
        (config.aux_weight_min_ratio * config.aux_weight),
        power=1.0
    )
    tb.summary.scalar("aux_weight", aux_weight)

    return Model(
        gen_model=gen_model,
        proposal_model=proposal_model,
        aux_model=aux_model,
        hamiltonian_is=hamiltonian_is,
        interleaving_scheduler=interleaving_scheduler,
        aux_weight=aux_weight
    )


def predict_and_quantify(
        PREDICTOR, STATES, SOURCES, GRAPH, params, config,
        dataset_transform=None, num_steps=5, num_samples=10):
    assert not (config.pred_ar_filtering and config.use_lookahead)
    predict_locs, predict_scales, _, log_weights = PREDICTOR(
        GRAPH, ExternalSequences(SOURCES.time_attrs, None),
        SOURCES.observations, num_steps, num_samples,
        every_step=config.pred_every_step, mode=SOURCES.mode,
        auto_regressive_filtering=config.pred_ar_filtering,
        initial_belief_states=STATES.belief_states,
        initial_latent_states=STATES.latent_states,
        initial_latent_histories=STATES.latent_histories
    )

    avg_predictions = metrics.reduce_avg(predict_locs)
    median_predictions = metrics.reduce_median(predict_locs)
    # wavg_predictions = metrics.reduce_wavg(predict_locs, log_weights)
    # mode_predictions = metrics.reduce_mode(predict_locs, predict_scales)
    # max_prob_predictions = metrics.reduce_max_prob(predict_locs, log_probs)

    labels = SOURCES.labels
    if not config.pred_every_step:
        labels = labels[-1, ...]

    with tf.control_dependencies([
        tf.assert_equal(tf.shape(avg_predictions), tf.shape(labels)),
        tf.assert_equal(tf.shape(median_predictions), tf.shape(labels))
    ]):
        l2_norm_errors = tf.math.sqrt(tf.math.reduce_sum(
            tf.math.squared_difference(avg_predictions, labels), axis=-1
        ))  # ([T, ]H, B, N)
    horizon_errors = tf.math.reduce_mean(l2_norm_errors, axis=[-2, -1])
    avg_error = tf.math.reduce_mean(l2_norm_errors)

    if dataset_transform is not None:
        inverse = dataset_transform.tf_inverse_transform
        predict_locs = inverse(predict_locs)
        avg_predictions = inverse(avg_predictions)
        median_predictions = inverse(median_predictions)
        labels = inverse(labels)

    error_metrics = dict()
    for metric, prediction in itertools.product(
        zip(
            ["MSE", "MAE", "MAPE"],
            [metrics.MSE, metrics.MAE, metrics.MAPE]
        ),
        zip(
            ["AVG", "MEDIAN"],
            [avg_predictions, median_predictions]
        )
    ):
        key = "{}_{}".format(prediction[0], metric[0])
        error_metrics[key] = metric[1](prediction[1], labels)

    predict_metrics = {
        "raw_horizon_errors": horizon_errors,
        "raw_avg_error": avg_error,
        "PICP": metrics.PICP(predict_locs, labels),
        **error_metrics
    }
    return predict_locs, predict_metrics


def make_vi_template(MODEL, params, config):
    assert not (config.train_obj == "iwae" and config.analytic_kl)
    ''' Variational Inference '''
    estimate, predict = VI(
        gen_model=MODEL.gen_model,
        inf_model=MODEL.proposal_model,
        aux_model=MODEL.aux_model,
        analytic_kl=config.analytic_kl,
        num_preview_steps=config.num_preview_steps,
        pred_resample_init=config.pred_resample_init,
        parallel_iterations=config.parallel_iterations,
        swap_memory=config.swap_memory
    )
    return Template(estimate, predict)


def make_vsmc_template(MODEL, params, config):
    ''' Variational Sequential Monte Carlo '''
    estimate, predict = VSMC(
        model=MODEL.gen_model,
        proposal=MODEL.proposal_model,
        aux_model=MODEL.aux_model,
        interleaving_scheduler=MODEL.interleaving_scheduler,
        # resample_jointly=True,
        resample_impl=config.vsmc_resample_impl,
        pred_resample_init=config.pred_resample_init,
        hamiltonian_is=MODEL.hamiltonian_is,
        analytic_kl=config.analytic_kl,
        parallel_iterations=config.parallel_iterations,
        swap_memory=config.swap_memory,
        summary_keys=[SUMMARY_TRAIN, SUMMARY_EVAL]
    )
    return Template(estimate, predict)


def build_vi_train_flow(MODEL, TEMPLATE, STATES, SOURCES, GRAPH,
                        params, config):
    anneal_factor = tf.constant(1.0)
    if config.kl_anneal:
        anneal_factor = 1.0 - tf.train.cosine_decay(
            1.0, tf.get_or_create_global_step(),
            params["num_steps"] // 2
        )
    tb.summary.scalar("anneal_factor", anneal_factor)

    EST, new_states = TEMPLATE.estimate(
        mode=SOURCES.mode, graph=GRAPH,
        observations=SOURCES.observations, inputs=SOURCES.time_attrs,
        num_samples=params["num_samples"], anneal_factor=anneal_factor,
        initial_belief_states=STATES.belief_states,
        initial_latent_states=STATES.latent_states,
        initial_latent_histories=STATES.latent_histories
    )
    EST.install_summaries([SUMMARY_TRAIN], "TRAIN")

    bound = EST.iwae_bound if config.train_obj == "iwae" else EST.vi_bound
    loss = tf.math.add_n([
        tf.math.negative(bound),
        MODEL.aux_weight * tf.math.negative(EST.aux_score),
        params["preview_loss_weight"] * EST.preview_divergence
    ])
    # TODO: Averaging IWAE bound may be problematic?
    # train_loss = tf.math.divide(loss, tf.to_float(
    #     tf.math.reduce_prod(tf.shape(SOURCES.observations)[:-1])  # T*B*N
    # ))
    return new_states, loss


def build_vi_eval_flow(MODEL, TEMPLATE, STATES, SOURCES, GRAPH,
                       params, config):
    EST, _ = TEMPLATE.estimate(
        mode=SOURCES.mode, graph=GRAPH,
        observations=SOURCES.observations,
        inputs=SOURCES.time_attrs,
        num_samples=params["eval_num_samples"],
        initial_belief_states=STATES.belief_states,
        initial_latent_states=STATES.latent_states,
        initial_latent_histories=STATES.latent_histories
    )
    EST.install_summaries([SUMMARY_EVAL], "EVAL")

    eval_metrics = {
        "ELBO": EST.vi_bound,
        "IWAE": EST.iwae_bound,
        "aux_score": EST.aux_score,
        "distortion": EST.distortion,
        "divergence": EST.divergence,
        "preview_divergence": EST.preview_divergence
    }
    return eval_metrics


def build_vsmc_train_flow(MODEL, TEMPLATE, STATES, SOURCES, GRAPH,
                          params, config):
    EST, new_states = TEMPLATE.estimate(
        mode=SOURCES.mode, graph=GRAPH,
        external=ExternalSequences(SOURCES.time_attrs, None),
        observations=SOURCES.observations,
        num_particles=params["num_samples"],
        initial_belief_states=STATES.belief_states,
        initial_latent_states=STATES.latent_states,
        initial_latent_histories=STATES.latent_histories
    )
    EST.install_summaries([SUMMARY_TRAIN], "TRAIN")
    loss = tf.math.add_n([
        tf.math.negative(EST.vsmc_bound),
        MODEL.aux_weight * tf.math.negative(EST.aux_score)
    ])
    # train_loss = tf.math.divide(loss, tf.to_float(
    #     tf.math.reduce_prod(tf.shape(SOURCES.observations)[:-1])  # T*B*N
    # ))
    return new_states, loss


def build_vsmc_eval_flow(MODEL, TEMPLATE, STATES, SOURCES, GRAPH,
                         params, config):
    EST, _ = TEMPLATE.estimate(
        mode=SOURCES.mode, graph=GRAPH,
        external=ExternalSequences(SOURCES.time_attrs, None),
        observations=SOURCES.observations,
        num_particles=params["eval_num_samples"],
        initial_belief_states=STATES.belief_states,
        initial_latent_states=STATES.latent_states,
        initial_latent_histories=STATES.latent_histories
    )
    EST.install_summaries([SUMMARY_EVAL], "EVAL")
    eval_metrics = {
        "aux_score": EST.aux_score,
        "NLL": tf.math.negative(EST.vsmc_bound)
    }
    return eval_metrics


def preprocess_graph(SOURCES, params, config):
    if config.embed_node_attr and SOURCES.node_attrs is not None:
        node_attrs = keras.layers.Dense(
            config.dim_node_embed, activation="tanh",
            input_shape=(params["dim_node_attrs"],)
        )(SOURCES.node_attrs)
    else:
        node_attrs = SOURCES.node_attrs

    if config.learn_node_embed:
        assert config.const_num_nodes is not None
        node_embeddings = tf.get_variable(
            "node_embeddings",
            shape=[config.const_num_nodes, config.dim_node_embed],
            trainable=True, initializer=tf.initializers.glorot_normal()
        )
        node_attrs = node_embeddings \
            if node_attrs is None \
            else util.broadcast_concat(node_attrs, node_embeddings)

    if config.embed_edge_attr and SOURCES.edge_attrs is not None:
        edge_attrs = keras.layers.Dense(
            config.dim_edge_embed, activation="tanh",
            input_shape=(params["dim_edge_attrs"],)
        )(SOURCES.edge_attrs)
    else:
        edge_attrs = SOURCES.edge_attrs

    return RuntimeGraph(
        edges=SOURCES.edges,
        node_mask=SOURCES.node_mask,
        center_mask=SOURCES.center_mask,
        edge_mask=SOURCES.edge_mask,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        dense=(config.gnn_impl == GNN_DENSE)
    )


def build_train_and_eval_flow(MODEL, STATES, SOURCES, params, config,
                              dataset_transform=None):
    GRAPH = preprocess_graph(SOURCES, params, config)

    if config.learn_init_states:
        tf.logging.info("Use trainable initial states.")
        STATES = STATES.trainable()

    obj = config.train_obj
    if obj == "vi" or obj == "iwae":
        make_template, train_flow, eval_flow = \
            make_vi_template, build_vi_train_flow, build_vi_eval_flow
    elif obj == "vsmc" and not config.mini_batch:
        make_template, train_flow, eval_flow = \
            make_vsmc_template, build_vsmc_train_flow, build_vsmc_eval_flow
    else:
        raise ValueError("Unknown/Invalid training objective.")

    TEMPLATE = make_template(MODEL, params, config)
    new_states, train_loss = train_flow(
        MODEL, TEMPLATE, STATES, SOURCES, GRAPH, params, config
    )
    eval_metrics = eval_flow(
        MODEL, TEMPLATE, STATES, SOURCES, GRAPH, params, config
    )
    quick_predict_ops, slow_predict_ops = [
        predict_and_quantify(
            TEMPLATE.predict, STATES, SOURCES, GRAPH, params, config,
            num_steps=num_steps, num_samples=num_samples,
            dataset_transform=dataset_transform
        )
        for num_steps, num_samples in zip(
            [config.train_num_pred_steps, config.eval_num_pred_steps],
            [config.train_num_pred_samples, config.eval_num_pred_samples]
        )
    ]
    quick_predictions, quick_predict_metrics = quick_predict_ops
    slow_predictions, slow_predict_metrics = slow_predict_ops
    return (
        new_states, train_loss, eval_metrics,
        quick_predictions, quick_predict_metrics,
        slow_predictions, slow_predict_metrics
    )


def install_summary_endpoints(train_loss, eval_metrics,
                              quick_predict_metrics, slow_predict_metrics):

    tf.summary.scalar("loss", train_loss,
                      collections=[SUMMARY_TRAIN], family="TRAIN")

    def install_predict_summaries(mode, predict_ops, key):
        tf.summary.scalar(
            mode + "/raw_avg_error", predict_ops["raw_avg_error"],
            collections=[key], family="PREDICT")
        tf.summary.histogram(
            mode + "/MSE", predict_ops["AVG_MSE"],
            collections=[key], family="PREDICT")
        tf.summary.histogram(
            mode + "/MAE", predict_ops["AVG_MAE"],
            collections=[key], family="PREDICT")
        tf.summary.histogram(
            mode + "/MAPE", predict_ops["AVG_MAPE"],
            collections=[key], family="PREDICT")

    install_predict_summaries(
        "quick", quick_predict_metrics, SUMMARY_QUICK_PREDICT)
    install_predict_summaries(
        "slow", slow_predict_metrics, SUMMARY_SLOW_PREDICT)


def save_predictions(quick_predictions, slow_predictions):
    tf.summary.tensor_summary(
        "quick_predictions", quick_predictions,
        collections=[SUMMARY_QUICK_PREDICT]
    )
    tf.summary.tensor_summary(
        "slow_predictions", slow_predictions,
        collections=[SUMMARY_SLOW_PREDICT]
    )


def build_tf_graph(
        MODEL, STATES, SOURCES, dataset_transform, params, config):
    global_step = tf.train.get_or_create_global_step()

    new_states, train_loss, eval_metrics, \
        quick_predictions, quick_predict_metrics, \
        slow_predictions, slow_predict_metrics = build_train_and_eval_flow(
            MODEL, STATES, SOURCES, params, config,
            dataset_transform=dataset_transform
        )
    install_summary_endpoints(
        train_loss, eval_metrics,
        quick_predict_metrics, slow_predict_metrics
    )
    # save_predictions(quick_predictions, slow_predictions)

    decay_steps = params["decay_steps"]
    if decay_steps is None:
        decay_steps = params["num_steps"]
    learning_rate = tf.train.noisy_linear_cosine_decay(
        learning_rate=params["learning_rate"],
        global_step=global_step,
        decay_steps=decay_steps,
        initial_variance=params["learning_rate_init_variance"],
        variance_decay=params["learning_rate_variance_decay"],
        beta=params["learning_rate_min_ratio"]
    )
    warmup_steps = max(1, params["learning_rate_warmup_steps"])
    warmup = tf.math.divide(
        tf.cast(global_step, tf.float32),
        tf.cast(warmup_steps, tf.float32)
    )
    learning_rate = tf.where(
        tf.math.less_equal(global_step, warmup_steps),
        tf.math.multiply(warmup, params["learning_rate"]),
        learning_rate
    )
    tb.summary.scalar("learning_rate", learning_rate)

    if config.optimizer is None or config.optimizer == "Adam":
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            epsilon=config.adam_eps
        )
    elif config.optimizer == "SGD":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError("Unknown optimizer: " + config.optimizer)

    if config.clip_gradient:
        threshold = params["clip_threshold"]
        gradients, variables = zip(*optimizer.compute_gradients(train_loss))
        global_norm = tf.linalg.global_norm(gradients)
        global_norm = tf.where(
            tf.math.logical_or(
                tf.math.is_inf(global_norm),
                tf.math.is_nan(global_norm)
            ),
            tf.constant(1E20), global_norm
        )
        gradients, global_norm = tf.clip_by_global_norm(
            gradients, threshold, use_norm=global_norm
        )
        tb.summary.scalar(
            "global_norm", global_norm, collections=[SUMMARY_TRAIN]
        )
        with tf.control_dependencies([tf.assign_add(global_step, 1)]):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
    else:
        train_op = optimizer.minimize(train_loss, global_step=global_step)

    return FlowSinks(
        train_op=train_op,
        new_states=new_states,
        train_loss=train_loss,
        eval_metrics=eval_metrics,
        quick_predictions=quick_predictions,
        quick_predict_metrics=quick_predict_metrics,
        slow_predictions=slow_predictions,
        slow_predict_metrics=slow_predict_metrics
    )
