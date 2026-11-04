# ====================================================================================
# Source: https://github.com/automl/TabPFN/blob/bef0cf257f3c4c519a2f8e301e0b81a26ae73aff/tabpfn/scripts/model_configs.py
# Mechanical namespace port of the TabPFN v1 prior (branch tabpfn_v1): only the
# lines listed in vendor/tabpfn1/PATCHES.md were changed. No algorithmic changes.
# fmt: off
# isort: off
# ====================================================================================
import torch


def get_diff_flex():
    """"
    Returns the configuration parameters for a differentiable wrapper around the tabular multiclass wrapper.
    """
    diff_flex = {
        # "ordinal_pct": {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        # "num_categorical_features_sampler_a": hp.choice('num_categorical_features_sampler_a',
        #                                                 [{'distribution': 'uniform', 'min': 0.3, 'max': 0.9}, None]),
        # "num_categorical_features_sampler_b": {'distribution': 'uniform', 'min': 0.3, 'max': 0.9},

        "output_multiclass_ordered_p": {'distribution': 'uniform', 'min': 0.0, 'max': 0.5}, #CSH.CategoricalHyperparameter('output_multiclass_ordered_p', [0.0, 0.1, 0.2]),
        "multiclass_type": {'distribution': 'meta_choice', 'choice_values': ['value', 'rank']},
    }

    return diff_flex

def get_diff_gp():
    """"
    Returns the configuration parameters for a differentiable wrapper around GP.
    """
    diff_gp = {
        'outputscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
                        'lower_bound': 0},
        'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
                        'lower_bound': 0},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]}
    }

    return diff_gp

def get_diff_causal():
    """"
    Returns the configuration parameters for a differentiable wrapper around MLP / Causal mixture.
    """
    diff_causal = {
        #"mix_activations": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        #"num_layers": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 6, 'min_mean': 1, 'round': True,
        #               'lower_bound': 2},
        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True,
                       'lower_bound': 2},
        # Better beta?
        #"prior_mlp_hidden_dim": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 130, 'min_mean': 5,
        #                         'round': True, 'lower_bound': 4},
        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},

        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
    # This mustn't be too high since activations get too large otherwise

        "noise_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': .3, 'min_mean': 0.0001, 'round': False,
                      'lower_bound': 0.0},
        "init_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.01, 'round': False,
                     'lower_bound': 0.0},
        #"num_causes": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 12, 'min_mean': 1, 'round': True,
        #               'lower_bound': 1},
        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                                 'lower_bound': 2},

        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
        "prior_mlp_activations": {'distribution': 'meta_choice_mixed', 'choice_values': [
            torch.nn.Tanh
            , torch.nn.Identity
            , torch.nn.ReLU
        ]},
        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        #'pre_sample_causes': {'distribution': 'meta_choice', 'choice_values': [True, False]},
    }

    return diff_causal

def get_diff_prior_bag():
    """"
    Returns the configuration parameters for a GP and MLP / Causal mixture.
    """
    diff_prior_bag = {
        'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0},
        # MLP Weight (Biased, since MLP works better, 1.0 is weight for prior number 0)
    }

    return diff_prior_bag

def get_diff_config():
    """"
    Returns the configuration parameters for a differentiable wrapper around GP and MLP / Causal mixture priors.
    """
    diff_prior_bag = get_diff_prior_bag()
    diff_causal = get_diff_causal()
    diff_gp = get_diff_gp()
    diff_flex = get_diff_flex()

    config_diff = {'differentiable_hyperparameters': {**diff_prior_bag, **diff_causal, **diff_gp, **diff_flex}}

    return config_diff
