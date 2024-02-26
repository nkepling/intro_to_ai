import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class BackProp():
    def __init__(self, input_size, hidden_size, output_size) -> None:
            """
            Initializes the neural network with the given input size, hidden size, and output size.
            
            Parameters:
            input_size (int): The size of the input layer.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer.
            """
            
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
        """
        Performs forward propagation through the neural network.

        Parameters:
        X (numpy.ndarray): Input data.

        Returns:
        dict: Dictionary containing the activations of the hidden and output layers.
        """
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
        """
        Backward propagation step of the neural network.

        Parameters:
        - X (numpy.ndarray): Input data of shape (m, n), where m is the number of samples and n is the number of features.
        - y (numpy.ndarray): Target labels of shape (m, 1).
        - activations (dict): Dictionary containing the activations of each layer.

        Returns:
        - None
        """
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
            """
            Trains the neural network using the given input data and labels.

            Parameters:
            - X (numpy.ndarray): Input data of shape (num_samples, num_features).
            - y (numpy.ndarray): Labels of shape (num_samples, num_classes).
            - epochs (int): Number of training epochs (default: 1000).
            - learning_rate (float): Learning rate for weight and bias updates (default: 0.01).

            Returns:
            - loss_history (list): List of loss values at each epoch.
            """
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
                
                # # Print loss every 100 epochs
                # if epoch % 100 == 0:
                #     print(f"Epoch {epoch}, Loss: {loss}")
            
            return loss_history

    def predict(self, X):
        """
        Predicts the output for the given input.

        Parameters:
        X (numpy.ndarray): The input data.

        Returns:
        numpy.ndarray: The predicted output.
        """
        activations = self.forward(X)
        return activations['a2']

    def mean_squared_error(self, y_true, y_pred):
        """
        Calculates the mean squared error between the true values and the predicted values.

        Parameters:
            y_true (array-like): The true values.
            y_pred (array-like): The predicted values.

        Returns:
            float: The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def sigmoid(self, x):
        """
        Compute the sigmoid function of the input.

        Parameters:
        x (float): The input value.

        Returns:
        float: The sigmoid value of the input.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Calculate the derivative of the sigmoid function.

        Parameters:
        x (float): The input value.

        Returns:
        float: The derivative of the sigmoid function.
        """
        return x * (1 - x)
    
    def plot_loss(self, loss):
        """
        Plots the loss history over epochs.

        Parameters:
        loss (list): List of loss values.

        Returns:
        None
        """
        plt.plot(loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History over Epochs')
        plt.show()


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

def gini_impurity(y):
    proportions = np.bincount(y) / len(y)
    return 1 - sum([p**2 for p in proportions if p > 0])

def calculate_best_split(X, y):
    best_feature, best_threshold = None, None
    best_gain = -1
    n_samples, n_features = X.shape
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_idx = np.where(X[:, feature] < threshold)
            right_idx = np.where(X[:, feature] >= threshold)
            if len(left_idx[0]) == 0 or len(right_idx[0]) == 0:
                continue
            
            left_impurity = gini_impurity(y[left_idx])
            right_impurity = gini_impurity(y[right_idx])
            gain = gini_impurity(y) - (len(left_idx[0]) / n_samples * left_impurity + len(right_idx[0]) / n_samples * right_impurity)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
                
    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=2):
    n_samples, n_features = X.shape
    num_labels = len(np.unique(y))
    
    # stopping criteria
    if depth >= max_depth or n_samples < 2 or num_labels == 1:
        leaf_value = np.argmax(np.bincount(y))
        return Node(value=leaf_value)
    
    feature, threshold = calculate_best_split(X, y)
    if feature is None:
        leaf_value = np.argmax(np.bincount(y))
        return Node(value=leaf_value)
    
    left_idx = np.where(X[:, feature] < threshold)
    right_idx = np.where(X[:, feature] >= threshold)
    left = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)
    return Node(feature, threshold, left, right)

def predict(sample, tree):
    while not tree.is_leaf_node():
        if sample[tree.feature] < tree.threshold:
            tree = tree.left
        else:
            tree = tree.right
    return tree.value






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
    # print(x_sample)
    # print(f"Predicted: {model.predict(x_sample)}, Actual: {y[sample_index]}")

    decision_tree_data = data_preprocessing_decision_tree(data)
    y = decision_tree_data['Label'].values

    #split the data into training and testing
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = build_tree(train_X, train_y, max_depth=15)

    #predictions = [predict(x, tree) for x in X]
    predictions = [predict(x, tree) for x in train_X]
    print("Train Accuracy")
    print((sum(predictions == train_y))/len(train_X))

    print("Test Accuracy")
    predictions = [predict(x, tree) for x in test_X]
    print((sum(predictions == test_y))/len(test_X))










    
