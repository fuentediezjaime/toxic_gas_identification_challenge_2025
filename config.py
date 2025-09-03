'''
Configuration file to input the dictionaries that define the search space (optimizable parameters) and the fixed parameters for the model,
as well as other control variables for the execution.
'''


#Optimizable model parameters
search_space_input = {
    'lgbm': {
        'n_estimators': {'type': 'int', 'range': [100, 1000]},
        'learning_rate': {'type': 'float', 'range': [0.01, 0.3], 'kwargs': {'log': True}}#,
#        'max_depth': {'type': 'int', 'range': [3, 10]},
#        'num_leaves': {'type': 'int', 'range': [20, 300]}
    },
    'svr': {
        'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf', 'poly']},
        'C': {'type': 'float', 'range': [1e-4, 1e4], 'kwargs': {'log': True}},
        'gamma': {'type': 'float', 'range': [1e-4, 1e2], 'kwargs': {'log': True}}
    }
}

# Fixed model parameters
fixed_parameters_input = {
    'lgbm': {},
    'svr': {}
}


# Main input parameters:
processor = 'cpu' #Keyword for the cpu/gpu execution of the trees
model_type = 'lgbm' #Keyword for the model type to be selected among the search space input
fold_n = 5
hyp_trials = 10