'''
Module containing functions to load data and split them, identify blocks, perform PCA on them
or other dimensionality reductions.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def load_and_separate_data(path_x : str, path_y : str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return x_train, x_test, y_train, y_test





def custom_cost(y_pred: np.array, y_true_pd: pd.DataFrame) -> float:
    y_true = y_true_pd.to_numpy() # Convert the true y values to numpy array.

    scale_factors = np.where(y_true < 0.5, 1.0, 1.2)  #If y_true small, 1, if y_true large, large scaling
    squared_errors = (y_true - y_pred)**2
    weighted_errors = scale_factors * squared_errors
    return np.mean(weighted_errors)