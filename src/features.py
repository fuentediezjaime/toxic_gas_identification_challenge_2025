'''
Module containing functions to load data and split them, identify blocks, perform PCA on them
or other dimensionality reductions.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .model import LgbmMultiout
from sklearn.model_selection import KFold

def load_and_separate_data(path_x : str, path_y : str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Reads the dfs containing the training set, removes the ID column for each df and 
    separates a training and a test dataset.
    Inputs:
        path_x: the path to the df containing the features of the dataset.
        path_y: the path to the df containing the targets of the dataset.

    Outputs:
        x_train: features of the training partition of the dataset
        x_test: features of the testing partition of the dataset
        y_train: targets of the training partition of the dataset
        y_test: targets of the testing partition of the dataset
    '''
    df_x = pd.read_csv(path_x).drop('ID', axis=1)
    df_y = pd.read_csv(path_y).drop('ID', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=17)
    try:
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
    except Exception as e:
        raise RuntimeError(f'Error converting train test splitted dataframes into numpy arrays: {e}')
    
    return x_train, x_test, y_train, y_test


def custom_cost(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    scale_factors = np.where(y_true < 0.5, 1.0, 1.2)  #If y_true small, 1, if y_true large, large scaling
    squared_errors = (y_true - y_pred)**2
    weighted_errors = scale_factors * squared_errors
    return np.mean(weighted_errors)



def get_param_suggestions(trial, param_ranges, fixed_params):
    variable_params = {}
    for par_name, par_features in param_ranges.items():
        match par_features['type']:
            case 'int': # we need to check the type of the parameter to use the adequate suggest method from the trial.
                v_low, v_high = par_features['range']
                par_suggestion = trial.suggest_int(f'{par_name}', v_low, v_high)
            
            case 'float':
                v_low, v_high = par_features['range']
                par_suggestion = trial.suggest_float(f'{par_name}', v_low, v_high)
                
            case 'categorical':
                possible_values = par_features['choices']
                par_suggestion = trial.suggest_categorical(f'{par_name}', possible_values)

        variable_params[par_name] = par_suggestion
    all_params={**fixed_params,**variable_params}
    return all_params
    


def objective(trial, X_data: np.array, Y_data: np.array, search_space: dict, fixed_space:dict, model_type: str, n_folds: int=3, device: str='cpu'):
    '''
    This is the key function for Optuna. It instantiates the model object, fits it and uses it to generate a prediction, which will be scored by
    the custom_cost function in features.py. Cross validation is used to define a score. 
    '''
    if model_type not in search_space.keys():
        raise ValueError('Model type not present in the parameter configuration dictionary')
     

    # Generate the instance of the model that we want
    match model_type:
        case 'lgbm':
            par_suggestions = get_param_suggestions(trial, search_space[model_type], fixed_space[model_type]) # Get params
            model = LgbmMultiout(model_params=par_suggestions, processor=device) #Instatiate model
        case 'xgboost':
            pass
        case 'random_forest':
            pass
        case 'ridge_reg':
            pass
        case 'fcnn_nostructure':
            pass
        case 'autoencoder':
            pass




    #With the model ready, go into the kfold
    kfolder = KFold(n_splits=n_folds)
    kf_splits = kfolder.split(X_data)
    error = []
    for kf_id, (train_ids, test_ids) in enumerate(kf_splits):
        x_train = X_data[train_ids]
        x_test = X_data[test_ids]
        y_train = Y_data[train_ids]
        y_test = Y_data[test_ids]

        model.train(x_train, y_train)
        y_pred = model.predict(x_test)
        error.append(custom_cost(y_pred, y_test))
    return np.average(error)