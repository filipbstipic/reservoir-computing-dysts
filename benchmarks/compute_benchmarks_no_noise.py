import sys
import os
import json

import dysts
from dysts.datasets import *
import collections

import pandas as pd
import numpy as np

module_paths = [
    os.path.abspath(os.getcwd()),
]

for module_path in module_paths:
    print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)

import models
from models.ESN_clean import ESN
from benchmarks.results.read_results import ResultsObject


import darts
from darts.models import *
from darts import TimeSeries
import darts.models


cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()
input_path = os.path.dirname(cwd)  + "/dysts/data/test_univariate__pts_per_period_100__periods_12.json"
## link to TEST data

dataname = os.path.splitext(os.path.basename(os.path.split(input_path)[-1]))[0]
output_path = cwd + "/results/results_" + dataname + ".json"
dataname = dataname.replace("test", "train" )
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

with open(hyperparameter_path, "r") as file:
    all_hyperparameters = json.load(file)

try:
    with open(output_path, "r") as file:
        all_results = json.load(file)
except FileNotFoundError:
    all_results = dict()
    

failed_combinations = collections.defaultdict(list)
METRIC = 'smape'
results_path = os.getcwd() + '/benchmarks/results/results_test_univariate__pts_per_period_100__periods_12.json'
results = ResultsObject(path=results_path)
results.sort_results(print_out=False, metric=METRIC)

for equation_name in equation_data.dataset:
    
    train_data = np.copy(np.array(equation_data.dataset[equation_name]["values"]))

    if equation_name not in all_results.keys():
        all_results[equation_name] = dict()
    
    split_point = int(5/6 * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, y_test_ts = TimeSeries.from_dataframe(pd.DataFrame(train_data)).split_before(split_point)
    
    all_results[equation_name]["values"] = np.squeeze(y_val)[:-1].tolist()
    
    for model_name in list(all_hyperparameters[equation_name].keys()) + ['ESN']:
        
        if model_name in all_results[equation_name].keys():
            continue
            
        all_results[equation_name][model_name] = dict()
        
        print(equation_name + " " + model_name, flush=True)
        
        if model_name == 'ESN':
            
            print(y_train.shape)
            
            model = ESN()
            model.fit(y_train)
            
            y_val_pred = model.predict(len(y_val))
            pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred)))
            true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))
            
            all_results[equation_name][model_name]["prediction"] = np.squeeze(y_val_pred).tolist()
            
        else:
            
            model = getattr(darts.models, model_name)(**all_hyperparameters[equation_name][model_name])
                                                            
            model.fit(y_train_ts)
            y_val_pred = model.predict(len(y_val))
            pred_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val_pred.values())))
            true_y = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(y_val)[:-1]))
        
            all_results[equation_name][model_name]["prediction"] = np.squeeze(y_val_pred.values()).tolist()
        
        
        for metric_name in metric_list:
            
            metric_func = getattr(darts.metrics.metrics, metric_name)
            score = metric_func(true_y, pred_y)
            #print(metric_name, score)
            if metric_name == METRIC:
                results.update_results(equation_name, model_name, score)

    # TODO: print ranking relative to others for that dynamical system
print('Failed combinations', failed_combinations)
results.get_average_rank(model_name, print_out=True)
            
            
            
            
            
            
            
            
            
            
            