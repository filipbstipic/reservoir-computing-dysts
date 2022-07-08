import argparse
import collections
import os
import random
import traceback
import warnings

import darts
import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
import json

from benchmarks.results.read_results import ResultsObject
from dysts.datasets import load_file


def set_seed(seed):
    seed %= 4294967294
    print(f'Using seed {seed}')
    random.seed(seed)
    np.random.seed(seed)


def eval_single_dyn_syst(model, dataset):
    cwd = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.dirname(cwd) + "/dysts/data/test_univariate__pts_per_period_100__periods_12.json"
    dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
    output_path = cwd + "/results/results_" + dataname + ".json"
    dataname = dataname.replace("test", "train")
    hyperparameter_path = cwd + "/hyperparameters/hyperparameters_" + dataname + ".json"
    metric_list = [
        'coefficient_of_variation',
        'mae',
        'mape',
        'marre',
        'mse',
        'r2_score',
        'rmse',
        'smape'
    ]
    equation_name = load_file(input_path).dataset[dataset]
    model_name = model.model_name
    failed_combinations = collections.defaultdict(list)
    METRIC = 'smape'
    results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_100__periods_12.json'
    results = ResultsObject(path=results_path)
    results.sort_results(print_out=False, metric=METRIC)

    train_data = np.copy(np.array(equation_name["values"]))

    split_point = int(5 / 6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

    try:
        model.fit(y_train_ts)
        y_val_pred = model.predict(len(y_val))
    except Exception as e:
        warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
        return np.inf
        failed_combinations[model_name].append(equation_name)
    pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
    true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

    print('-----', dataset, y_train_ts.values().shape)
    value = None
    for metric_name in metric_list:
        metric_func = getattr(darts.metrics.metrics, metric_name)
        score = metric_func(true_y, pred_y)
        print(metric_name, score)
        if metric_name == METRIC:
            value = score
            rank = results.update_results(dataset, model_name, score)
    return value, rank  # , model._cell.W_h



def eval_all_dyn_syst(model):
    cwd = os.path.dirname(os.path.realpath(__file__))
    # cwd = os.getcwd()
    input_path = os.path.dirname(cwd) + "/dysts/data/test_univariate__pts_per_period_100__periods_12.json"
    dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
    output_path = cwd + "/results/results_" + dataname + ".json"
    dataname = dataname.replace("test", "train")
    hyperparameter_path = cwd + "/hyperparameters/hyperparameters_" + dataname + ".json"
    metric_list = [
        'coefficient_of_variation',
        'mae',
        'mape',
        'marre',
        'mse',
        'r2_score',
        'rmse',
        'smape'
    ]
    equation_data = load_file(input_path)
    model_name = model.model_name
    failed_combinations = collections.defaultdict(list)
    METRIC = 'smape'
    results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_100__periods_12.json'
    results = ResultsObject(path=results_path)
    results.sort_results(print_out=False, metric=METRIC)
    for equation_name in equation_data.dataset:

        train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

        split_point = int(5 / 6 * len(train_data))
        y_train, y_val = train_data[:split_point], train_data[split_point:]
        y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

        try:

            model.fit(y_train_ts)
            y_val_pred = model.predict(len(y_val))

        except Exception as e:
            warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
            failed_combinations[model_name].append(equation_name)
            traceback.print_exc()
            continue

        pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
        true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))

        print('-----', equation_name)
        for metric_name in metric_list:
            metric_func = getattr(darts.metrics.metrics, metric_name)
            score = metric_func(true_y, pred_y)
            print(metric_name, score)
            if metric_name == METRIC:
                results.update_results(equation_name, model_name, score)

        # TODO: print ranking relative to others for that dynamical system
    print('Failed combinations', failed_combinations)
    results.get_average_rank(model_name, print_out=True)
    
    
def eval_all_dyn_syst_best_hyperparams(model, pts_per_period=100, save_results=False):
    
    cwd = os.path.dirname(os.path.realpath(__file__))
    
    input_path = os.path.dirname(cwd) + "/dysts/data/test_univariate__pts_per_period_" + str(pts_per_period)+ "__periods_12.json"
    dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
    output_path = cwd + "/results/results_" + dataname + ".json"
    dataname = dataname.replace("test", "train")
    
    hyperparameter_path = os.getcwd() + "/benchmarks/hyperparameters/hyperparameters_" + dataname + "_ESN.json"
    
    hyperparameters_file = open(hyperparameter_path)
    hyperparameters = json.load(hyperparameters_file)
    
    metric_list = [
        'coefficient_of_variation',
        'mae',
        'mape',
        'marre',
        'mse',
        'r2_score',
        'rmse',
        'smape'
    ]
    equation_data = load_file(input_path)
    failed_combinations = collections.defaultdict(list)
    METRIC = 'smape'
    model_name = model.model_name
    
    results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_' + str(pts_per_period)+ '__periods_12.json'
    out_results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_' + str(pts_per_period)+ '__periods_12_ESN.json'
    
    with open(results_path, "r") as file:
        all_results = json.load(file)
        
    results = ResultsObject(path=results_path)
    results.sort_results(print_out=False, metric=METRIC)
    
    print("Input path ", input_path)
    print("Hyperparameter path ", hyperparameter_path)
    print("Results path ", results_path)
    
    for equation_name in equation_data.dataset:
        
        if equation_name == "AthmosfericRegime":
            
            continue
 
        train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

        split_point = int(5 / 6 * len(train_data))
        y_train, y_val = train_data[:split_point], train_data[split_point:]
        y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)

        try:

            all_results[equation_name][model_name] = dict()

            model.set_hyperparams(hyperparameters[equation_name]['ESN'])

            print("hyperparams are", hyperparameters[equation_name]['ESN'])

            model.fit(y_train_ts)
            y_val_pred = model.predict(len(y_val))
            
        except Exception as e:
            warnings.warn(f'Could not evaluate {equation_name} for {model_name} {e.args}')
            failed_combinations[model_name].append(equation_name)
            traceback.print_exc()
            continue

        pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
        true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))
        all_results[equation_name][model_name]["prediction"] = np.squeeze(y_val_pred.values()).tolist()

        print('-----', equation_name)
        for metric_name in metric_list:
            metric_func = getattr(darts.metrics.metrics, metric_name)
            score = metric_func(true_y, pred_y)
            
            all_results[equation_name][model_name][metric_name] = score
            
            print(metric_name, score)
            if metric_name == METRIC:
                results.update_results(equation_name, model_name, score)

        # TODO: print ranking relative to others for that dynamical system
    print('Failed combinations', failed_combinations)
    
    results.get_average_rank(model_name, print_out=True)
    
    if save_results:
        
        with open(out_results_path, 'w') as fp:
            json.dump(all_results, fp, indent=4)

    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_hyperparam_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparam_config", help="See hyperparameter_config.py", type=str)
    parser.add_argument("--test_single_config", help="", type=str2bool, default=False)
    parser.add_argument("--pts_per_period", help="", type=int, default=100)
    args = parser.parse_args()
    return args



def getNewESNParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoir_size", help="reservoir_size", type=int, default=1000)
    parser.add_argument("--sparsity", help="sparsity", type=float, default=0.1)
    parser.add_argument("--radius", help="radius", type=float, default=0.95)
    parser.add_argument("--reg", help="regularization", type=float, default=1e-7)
    parser.add_argument("--alpha", help="alpha", type=float, default=1.0)
    
    #parser.add_argument("--seed", type=int, default=10)
    # parser.add_argument("--resample", type=str2bool, default=False)

    return parser


def new_args_dict():
    parser = getNewESNParser()
    args = parser.parse_args()

    args_dict = args.__dict__
    return args_dict


'''
y_val_pred = model.get_best_sig_pred(pred_list, len(y_val))

while y_val_pred == None:

    all_results[equation_name][model_name] = dict()

    if hyperparameters[equation_name]['ESN']['radius'] == 0.5:

        hyperparameters[equation_name]['ESN']['radius'] = 0.75

    else:

        hyperparameters[equation_name]['ESN']['radius'] += 0.1
        hyperparameters[equation_name]['ESN']['reg'] = 0.0001

    model.set_hyperparams(hyperparameters[equation_name]['ESN'])

    print("hyperparams are", hyperparameters[equation_name]['ESN'])

    pred_list = []

    for i in range(10):

        model.fit(y_train_ts)
        y_val_pred = model.predict(len(y_val) + 800)
        pred_list.append(y_val_pred)

    y_val_pred = model.get_best_sig_pred(pred_list, len(y_val))
'''
