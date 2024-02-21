import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class BackProp():
    def __init__(self, input_size, hidden_size, output_size) -> None:
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


        np.random.seed(42)  # For reproducibility

        self.weights = {
            'w1': np.random.randn(input_size, hidden_size),
            'w2': np.random.randn(hidden_size, output_size)
        }
        self.biases = {
            'b1': np.zeros((1, hidden_size)),
            'b2': np.zeros((1, output_size))
        }

            

    def forward(self, X):
        activations = {}
        # Input layer to hidden layer
        z1 = np.dot(X, self.weights['w1']) + self.biases['b1']
        a1 = self.sigmoid(z1)
        
        # Hidden layer to output layer
        z2 = np.dot(a1, self.weights['w2']) + self.biases['b2']
        a2 = self.sigmoid(z2)
        
        activations['a1'], activations['a2'] = a1, a2
        return activations


    def backward(self, X, y, activations):
        m = X.shape[0]
        deltas = {}
        
        # Calculate difference between actual value and predicted value
        error = activations['a2'] - y
        deltas['d2'] = error * self.sigmoid_derivative(activations['a2'])
        
        # Propagate the error backward
        error = np.dot(deltas['d2'], self.weights['w2'].T)
        deltas['d1'] = error * self.sigmoid_derivative(activations['a1'])
        
        # Update weights and biases
        self.weights['w2'] -= np.dot(activations['a1'].T, deltas['d2']) / m
        self.biases['b2'] -= np.sum(deltas['d2'], axis=0, keepdims=True) / m
        
        self.weights['w1'] -= np.dot(X.T, deltas['d1']) / m
        self.biases['b1'] -= np.sum(deltas['d1'], axis=0, keepdims=True) / m
        

    def train_neural_network(self, X, y, epochs=1000, learning_rate=0.01):
        loss_history = []
        
        for epoch in range(epochs):
            # Forward propagation
            activations = self.forward(X)
            
            # Compute loss
            loss = self.mean_squared_error(activations['a2'], y)
            loss_history.append(loss)
            
            # Backward propagation and update weights and biases
            
            # Update weights and biases with learning rate
            self.weights['w1'] -= learning_rate * self.weights['w1']
            self.weights['w2'] -= learning_rate * self.weights['w2']
            self.biases['b1'] -= learning_rate * self.biases['b1']
            self.biases['b2'] -= learning_rate * self.biases['b2']
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        
        return loss_history

    def predict(self, X):
        activations = self.forward(X)
        return activations['a2']

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def plot_loss(self, loss):
        plt.plot(loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History over Epochs')
        plt.show()



class NN_Classifier():

    def __init__(self) -> None:
        pass

    def forward(self):
        pass


def load_data(file):
    data = pd.read_csv(file, encoding='utf-16', delimiter='\t')
    return data

def data_preprocessing_decision_tree(data):
    data['Label'] = pd.cut(data['utility'], bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf], labels=[1, 2, 3, 4, 5])
    return data



def decision_tree():
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
    X = data.drop('utility', axis=1).values.astype(np.float32)
    y = data['utility'].values.astype(np.float32).reshape(-1, 1)

    model = BackProp(X.shape[1], 4, 1)
    loss = model.train_neural_network(X, y, epochs=200)

    #test one sample input
    sample_index = 176
    x_sample = X[sample_index]
    print(x_sample)
    print(f"Predicted: {model.predict(x_sample)}, Actual: {y[sample_index]}")









    
