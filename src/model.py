'''
Module containing optimizer callbacks and different model classes
'''

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os
from typing import List, Dict, Any, Tuple


class LgbmMultiout:
    def __init__(self, model_params: Dict[str, Any]=None, processor: str='cpu', npar: int=1):
        '''
        model_params: a dictionary containing lgbm model parameters for the instance.
        processor: a string indicating if the model runs in cpu or gpu.
        npar: the number of parallel threads to launch for the model. Note that this is parallelism at the model level,
        not at the cross validation or hyperparameter search level (can lead to nested threading and slow runs)
        '''
        self.model = MultiOutputRegressor(lgb.LGBMRegressor(**model_params or {}, device_type=processor),n_jobs=npar) #Initialize the model
        self._is_trained = False #Initialize the model as untrained

    def train(self, X_train: pd.DataFrame, Y_train: pd.DataFrame) -> None:
        self.model.fit(X_train, Y_train)
        self._is_trained = True #Now it is trained


    def predict(self, X_test: pd.DataFrame) -> np.array:
        if self._is_trained:
            return self.model.predict(X_test)
        else:
            raise RuntimeError('The model is not trained yet')
        
    @classmethod #So that we don't create a dummy before loading, just load and it creates.
    def load(self, path_load: str) -> 'LgbmMultiout':
        if os.path.exists(path_load):
            predictor = joblib.load(path_load)
            print('Loaded predictor')
            return predictor
        else:
            raise RuntimeError('The model to be loaded does not exist')

    def save(self, path_save: str):
        if self._is_trained:
            joblib.dump(self, path_save)
        else:
            raise RuntimeError('Model was not trained before saving')
        

