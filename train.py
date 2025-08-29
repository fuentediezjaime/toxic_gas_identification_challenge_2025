'''
This is the main file of the project, it reads the config file, which contains the model type, the calculation type and the
ranges for the hyperparameter range.
'''
from src.features import load_and_separate_data
def main():
    x_path = 'dataset/x_train_T9QMMVq.csv'
    y_path = 'dataset/y_train_R0MqWmu.csv'

    # Do the train test split
    x_train, x_test, y_train, y_test = load_and_separate_data(x_path, y_path)


    

if __name__ =='__main__':
    main()