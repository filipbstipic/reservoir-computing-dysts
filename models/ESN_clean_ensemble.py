import time

import darts
import numpy as np
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts import TimeSeries
import os
import sys

module_paths = [
    os.path.abspath(os.getcwd()),
]

for module_path in module_paths:
    print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from scipy import linalg
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.stats import rv_continuous
from numpy.random import default_rng
from scipy import stats

from models.utils import set_seed


import pandas as pd
from functools import partial

print = partial(print, flush=True)

from sklearn.linear_model import Ridge
from typing import Union, Sequence, Optional


class ESN(GlobalForecastingModel, BaseEstimator):
   
    def __init__(self, reservoir_size=1000, sparsity=0.1, radius=0.95, reg=1e-7, alpha=1.0, burn_in_ratio=0.2, model_name='ESN', seed=None, W_scaling=1, flip_sign=False, ensemble_size=5):

        
        self.radius = radius
        self.reservoir_size = reservoir_size
        self.radius = radius
        self.sparsity = sparsity
        self.alpha = alpha
        self.reg = reg
        self.burn_in_ratio = burn_in_ratio
        
        self.model_name = model_name
        self.W_scaling = W_scaling
        self.flip_sign = flip_sign
        
        self.inSize = 1
        self.outSize = 1
        
        self.scaler = StandardScaler() # I have found that scaling is quite important for performance, for simplicity I just standarize the input time series. When I predict I invert this transform. 

        self.sampleWeights()
        self._estimator_type = 'regressor'
        
        if seed != None:
            set_seed(seed)

    def sampleWeights(self):
        
        self.W = self.getHiddenMatrix()
        self.Win = (np.random.rand(self.reservoir_size,1 + self.inSize) - 0.5) * self.W_scaling # TODO I have read that in the literature that Win scaling is important, hence some other ways of initializing W_in could be considered. 
        
    
    def getHiddenMatrix(self):
        
        success = False
        counter = 3
        
        while not success and counter>=0:

            try:

                W = sparse.random(self.reservoir_size, self.reservoir_size, density=self.sparsity)
                W *= self.W_scaling 

                eigenvalues, eigvectors = splinalg.eigs(W)
                eigenvalues = np.abs(eigenvalues)
                W = (W / np.max(eigenvalues)) * self.radius

                success = True

                if self.flip_sign:

                    W *= (np.random.binomial(1, 0.5, (self.reservoir_size, self.reservoir_size)) - 0.5)*2

            except:

                sucess = False
                counter -= 1

        if counter < 0:

            print("-------------------- BIG CONVERGENCE PROBLEMS -----------------")

        return W
     
        
    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None
            ) -> None:
        
        # since fit will be called multiple times from the main for all dynamical systems, I resample the weights each time
        self.sampleWeights()
        
        series = np.squeeze(series.values())
        series = series.reshape(-1, 1)
        self.scaler.fit(series)
        series = self.scaler.transform(series)
        
        trainLen = series.shape[0] - 1
        self.total_trainLen = series.shape[0]
        
        initLen = int(trainLen*self.burn_in_ratio) # how many timestep you want to let the chain go through before you starting collecting states to fit the readout

        target_seq = series[initLen + 1:trainLen + 1] # target is simply the same series shifted by one timestep
        
        X = np.zeros((1 + self.inSize + self.reservoir_size, trainLen - initLen)) # Collected reservor states matrix (cols)
        Yt = target_seq.reshape(1, -1) 
        
        #x = np.random.rand(self.reservoir_size, 1) #TODO, if you want the initial reservoir state to be randomly chosen
        x = np.zeros((self.reservoir_size, 1))
        
        for t in range(trainLen + 1):

            u = series[t][0]
            x = (1 - self.alpha) * x + self.alpha * np.tanh( np.dot( self.Win, np.vstack((1, u)) ) + self.W @ x )
            if t >= initLen and t < trainLen:

                X[:,t - initLen] = np.vstack((1, u, x))[:,0]

        self.last_u = u.copy()
        self.last_x = x.copy()
        
        
        self.Wout = linalg.solve( np.dot(X,X.T) + self.reg*np.eye(1+self.inSize+self.reservoir_size), np.dot(X,Yt.T) ).T
        


    def predict(self,
                testLen: int,
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                num_samples: int = 1,
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        
        x = self.last_x
        u = self.last_u
        Y = np.zeros(testLen)
        
        for t in range(testLen):
            
            u = np.dot(self.Wout, np.vstack((1, u, x)))
            Y[t] = u
            x = (1 - self.alpha) * x + self.alpha*np.tanh( np.dot( self.Win, np.vstack((1, u)) ) + self.W @ x )
        
        Y = Y.reshape(-1, 1)
        Y_inverse_transform = self.scaler.inverse_transform(Y)
        
        df = pd.DataFrame(np.squeeze(Y_inverse_transform))
        df.index = range(self.total_trainLen, self.total_trainLen + testLen)
        predict_ts = TimeSeries.from_dataframe(df)
                
        return np.squeeze(Y_inverse_transform)
        
        '''
        Y_timeseries = TimeSeries.from_dataframe(pd.DataFrame(Y_inverse_transform))
        
        return Y_timeseries
        '''
    
    def set_hyperparams(self, hyperparams):
        
        self.reservoir_size = hyperparams['reservoir_size']
        self.sparsity = hyperparams['sparsity']
        self.radius = hyperparams['radius']
        self.reg = hyperparams['reg']
        self.alpha = hyperparams['alpha']
        
        
    def delete(self):
        return 0
