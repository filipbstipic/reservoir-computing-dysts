import copy

from models.utils import get_hyperparam_parser

args = get_hyperparam_parser()

ESN_hyperparam = {
    "reservoir_size": [250, 500, 1000],
    "sparsity": [0.1, 0.5], 
    "radius": [0.5, 0.75, 0.95],
    "reg": [1e-7, 1e-5],
    "alpha": [0.5, 0.75, 1.0],
    "burn_in_ratio": [0.2]
}

network_outputs = [1, 4]
network_outputs = [1]
pts_per_period = args.pts_per_period
network_inputs = [5, 10, int(0.5 * pts_per_period), pts_per_period]  # can't have kernel less than 5

# LSTM already evaluated
rnn_rnn_hyperparam = {
    "input_chunk_length": network_inputs,
    "output_chunk_length": network_outputs,
    "model": ["RNN"],
    "n_rnn_layers": [2],
    "n_epochs": [200]
}
rnn_gru_hyperparam = copy.deepcopy(rnn_rnn_hyperparam)
rnn_gru_hyperparam["model"] = ["GRU"]


def get_single_config(hyperparam_dict):
    for k, v in hyperparam_dict.items():
        hyperparam_dict[k] = [v[0]]

    return hyperparam_dict


hyperparameter_configs = {
    'ESN': ESN_hyperparam,
    'RNNModel_RNN': rnn_rnn_hyperparam,
    'RNNModel_GRU': rnn_gru_hyperparam
}
