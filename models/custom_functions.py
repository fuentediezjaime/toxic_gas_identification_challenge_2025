import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


def loss_custom(y_real, y_pred):

    if (y_real >= 0.5):
        error = 1.2*(y_real-y_pred)**2
    else: 
        error = 1*(y_real-y_pred)**2


def hyperpar_search(model, param_dict, n_cv):

    scorer = make_scorer(loss_custom,greater_is_better=False)
    # First, perfomr a random search
    rand_search = RandomizedSearchCV(estimator=model, 
                                     param_distributions=param_dict, 
                                     scoring=None, 
                                     cv=n_cv,
                                     n_iter=10)
    return rand_search