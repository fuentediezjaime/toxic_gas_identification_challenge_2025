'''
This is the main file of the project, it reads the config file, which contains the model type, the calculation type and the
ranges for the hyperparameter range.
'''
from src.features import load_and_separate_data
from src.features import objective
import optuna
from config import search_space_input, fixed_parameters_input, processor, model_type, hyp_trials, fold_n
def main():
    x_path = 'dataset/x_train_T9QMMVq.csv'
    y_path = 'dataset/y_train_R0MqWmu.csv'

    # Do the train test split
    x_train, x_test, y_train, y_test = load_and_separate_data(x_path, y_path)
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial:
             objective(trial,  x_train, y_train, search_space=search_space_input, 
                       fixed_space=fixed_parameters_input, model_type=model_type, 
                       n_folds=fold_n, device=processor),

        n_trials=hyp_trials)

if __name__ =='__main__':
    main()