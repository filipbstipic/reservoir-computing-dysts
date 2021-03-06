import time

import numpy as np
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

class ESN:
   
    def __init__(self, reservoir_size=1000, sparsity=0.1, radius=0.95, reg=1e-7, alpha=1.0, initLen=250, initialize_reservoir_0=True, model_name='ESN_clean', seed=None, W_scaling=1, flip_sign=False):

        
        self.radius = radius
        self.reservoir_size = reservoir_size
        self.radius = radius
        self.sparsity = sparsity
        self.alpha = alpha
        self.reg = reg
        
        self.model_name = model_name
        self.W_scaling = W_scaling
        
        self.inSize = 1
        self.outSize = 1
        self.initLen = initLen
        self.initialize_reservoir_0 = initialize_reservoir_0
        
        self.scaler = StandardScaler() # I have found that scaling is quite important for performance, for simplicity I just standarize the input time series. When I predict I invert this transform. 

        #self.sampleWeights()
        
        if seed != None:
            set_seed(seed)

    def sampleWeights(self):
        
        #self.W = self.getHiddenMatrix()
        self.W = np.random.rand(self.reservoir_size,self.reservoir_size) - 0.5
        rhoW = max(abs(linalg.eig(self.W)[0]))
        self.W *= self.radius / rhoW
        
        self.Win = (np.random.rand(self.reservoir_size,1 + self.inSize) - 0.5) * self.W_scaling 

    def getHiddenMatrix(self):
        
        success = False
        counter = 3
        
        while not success and counter>=0:

            try:

                W = sparse.random(self.reservoir_size, self.reservoir_size, density=self.sparsity)
                eigenvalues, eigvectors = splinalg.eigs(W)
                eigenvalues = np.abs(eigenvalues)
                W = (W / np.max(eigenvalues)) * self.radius

                success = True

            except:

                sucess = False
                counter -= 1

        if counter < 0:

            print("-------------------- BIG CONVERGENCE PROBLEMS -----------------")

        return W
     
    
    # series is the timeseries array of single measurements
    def fit(self, series):
        
        # since fit will be called multiple times from the main for all dynamical systems, I resample the weights each time
        self.sampleWeights()
        
        series = series.reshape(-1, 1)
        self.scaler.fit(series)
        series = self.scaler.transform(series)
        
        trainLen = series.shape[0] - 1 

        target_seq = series[self.initLen + 1:trainLen + 1] # target is simply the same series shifted by one timestep
        
        X = np.zeros((1 + self.inSize + self.reservoir_size, trainLen - self.initLen)) # Collected reservor states matrix (cols)
        Yt = target_seq.reshape(1, -1) 
        
        if not self.initialize_reservoir_0:
            x = np.random.rand(self.reservoir_size, 1) 
        else:
            x = np.zeros((self.reservoir_size, 1))
        
        for t in range(trainLen + 1):

            u = series[t][0]
            x = (1 - self.alpha) * x + self.alpha * np.tanh( np.dot( self.Win, np.vstack((1, u)) ) + self.W @ x )
            if t >= self.initLen and t < trainLen:

                X[:,t - self.initLen] = np.vstack((1, u, x))[:,0]

        self.last_u = u.copy()
        self.last_x = x.copy()
        
        self.Wout = linalg.solve( np.dot(X,X.T) + self.reg*np.eye(1 + self.inSize + self.reservoir_size), np.dot(X,Yt.T) ).T
    
    
    def predict(self, testLen, teacher_forcing_series=None):
        
        x = self.last_x
        u = self.last_u
        Y = np.zeros(testLen)

        X_test = np.zeros((self.reservoir_size, testLen))
                
        for t in range(testLen):
            
            u = np.dot(self.Wout, np.vstack((1, u, x)))
            Y[t] = u
            
            X_test[:,t] = x.T
            x = (1 - self.alpha) * x + self.alpha*np.tanh( np.dot( self.Win, np.vstack((1, u)) ) + self.W @ x )
            
        
        self.X_test = X_test
        Y = np.array(Y).reshape(-1, 1)
                
        return self.scaler.inverse_transform(Y)
    
 
        
    def delete(self):
        return 0
