import pandas as pd
import numpy as np


class NN_Classifier():

    def __init__(self) -> None:
        pass

    def forward(self):
        pass


def load_data(file_path) -> pd.DataFrame:
    data = pd.read_csv(file, encoding='utf-16', delimiter='\t')
    data['Label'] = pd.cut(data['utility'], bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf], labels=[1, 2, 3, 4, 5])
    return data



def decision_tree():
    pass


def backprop():
    pass

def MSE():
    pass

def cross_validation(X,y,k):
    """
    Overview:
        Randomaize the data and perform k fold cross-validation N times. 
        (ie. do 5 fold cv 10 times):

        Randomize data
            Partition data into k bins
            for b in bins:
                treat b at test set
                train on bins-b bins
                test on b. 
    Arguments:
        X : pd.DataFrame: Features
        y ; pd.DataFrame: labels
        k : int: number of 
    
    Returns: 
        Err: float: Average Acc or MSE of N runs of k-cv 
        Var: float: varainec of Accs over N runs of k-cv
    """
    pass


def main():
    pass




if  __name__ == "__main__":
    file = 'data.txt'
    data = load_data(file)
    print(data.head)



    
