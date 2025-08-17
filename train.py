'''
This is the main file of the project, it reads the config file, which contains the model type, the calculation type and the
ranges for the hyperparameter range.
'''
from src.features import load_and_separate_data
def main():
    x_path = 'dataset/x_train_T9QMMVq.csv'
    y_path = 'dataset/y_train_R0MqWmu.csv'
    x_tr, x_te, y_tr, y_te = load_and_separate_data(x_path, y_path)

    print(x_tr, x_te, y_tr, y_te)

if __name__ =='__main__':
    main()