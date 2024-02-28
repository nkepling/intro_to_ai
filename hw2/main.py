import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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


    def backward(self, X, y, activations, learning_rate=0.01):
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
        
        # Output layer error
        error = activations['a2'] - y
        deltas['d2'] = error * self.sigmoid_derivative(activations['a2'])
        
        # Hidden layer error
        error_hidden = np.dot(deltas['d2'], self.weights['w2'].T)
        deltas['d1'] = error_hidden * self.sigmoid_derivative(activations['a1'])
        
        # Update weights and biases
        self.weights['w2'] -= learning_rate * np.dot(activations['a1'].T, deltas['d2']) / m
        self.biases['b2'] -= learning_rate * np.sum(deltas['d2'], axis=0, keepdims=True) / m
        
        self.weights['w1'] -= learning_rate * np.dot(X.T, deltas['d1']) / m
        self.biases['b1'] -= learning_rate * np.sum(deltas['d1'], axis=0, keepdims=True) / m

    def train_neural_network(self, X, y, epochs=10000, learning_rate=0.01, loss_threshold=0.000001):
        """
        Trains the neural network using the given input data and labels.

        Parameters:
        - X (numpy.ndarray): Input data of shape (num_samples, num_features).
        - y (numpy.ndarray): Labels of shape (num_samples, num_classes).
        - epochs (int): Number of training epochs (default: 1000).
        - learning_rate (float): Learning rate for weight and bias updates (default: 0.01).
        - loss_threshold (float): Threshold for change in loss between consecutive epochs (default: 0.001).

        Returns:
        - loss_history (list): List of loss values at each epoch.
        """
        loss_history = []
        prev_loss = float('inf')  # Initialize previous loss with infinity
        
        for epoch in range(epochs):
            # Forward propagation
            activations = self.forward(X)
            
            # Compute loss
            loss = self.mean_squared_error(activations['a2'], y)
            loss_history.append(loss)

            # Break if loss is below the threshold
            if abs(prev_loss - loss) < loss_threshold:
                break
        
            
            # Update weights and biases with learning rate
            self.backward(X, y, activations, learning_rate=learning_rate)
            
            prev_loss = loss
            
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

class decision_tree():
    class Node:
        """
        Represents a node in a decision tree.

        Attributes:
            feature: The feature used for splitting at this node.
            threshold: The threshold value used for splitting at this node.
            left: The left child node.
            right: The right child node.
            value: The predicted value at this node (only applicable for leaf nodes).
        """

        def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
            """
            Initialize a node in a decision tree.

            Args:
                feature (str): The feature used for splitting at this node.
                threshold (float): The threshold value for the feature.
                left (Node): The left child node.
                right (Node): The right child node.
                value: The predicted value at this node (for leaf nodes).

            Returns:
                None
            """
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def is_leaf_node(self):
            """
            Checks if the node is a leaf node.

            Returns:
                True if the node is a leaf node, False otherwise.
            """
            return self.value is not None
        
    def __init__(self):
        """
        Initializes the class object.
        """
        self.tree = None


    def gini_impurity(self, y):
        """
        Calculate the Gini impurity of a given set of labels.

        Parameters:
        - y: A list or array of labels.

        Returns:
        - The Gini impurity value.

        """
        proportions = np.bincount(y) / len(y)
        return 1 - sum([p**2 for p in proportions if p > 0])

    def calculate_best_split(self, X, y):
            """
            Calculates the best feature and threshold to split the data based on the Gini impurity criterion.

            Parameters:
            X (numpy.ndarray): The input features.
            y (numpy.ndarray): The target labels.

            Returns:
            best_feature (int): The index of the best feature to split on.
            best_threshold (float): The threshold value for the best split.
            """
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
                    
                    left_impurity = self.gini_impurity(y[left_idx])
                    right_impurity = self.gini_impurity(y[right_idx])
                    gain = self.gini_impurity(y) - (len(left_idx[0]) / n_samples * left_impurity + len(right_idx[0]) / n_samples * right_impurity)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold
                        
            return best_feature, best_threshold

    def build_tree(self, X, y, depth=0, max_depth=4):
            """
            Builds a decision tree recursively using the given dataset.

            Parameters:
            - X: numpy array, shape (n_samples, n_features)
                The input features of the dataset.
            - y: numpy array, shape (n_samples,)
                The target labels of the dataset.
            - depth: int, optional (default=0)
                The current depth of the tree.
            - max_depth: int, optional (default=4)
                The maximum depth of the tree.

            Returns:
            - Node object
                The root node of the decision tree.
            """
            n_samples, n_features = X.shape
            num_labels = len(np.unique(y))
            
            # stopping criteria
            if depth >= max_depth or n_samples < 2 or num_labels == 1:
                leaf_value = np.argmax(np.bincount(y))
                return self.Node(value=leaf_value)
            
            feature, threshold = self.calculate_best_split(X, y)
            if feature is None:
                leaf_value = np.argmax(np.bincount(y))
                return self.Node(value=leaf_value)
            
            left_idx = np.where(X[:, feature] < threshold)
            right_idx = np.where(X[:, feature] >= threshold)
            left = self.build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
            right = self.build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)
            return self.Node(feature, threshold, left, right)
    
    def fit(self, X, y, max_depth):
        """
        Fits the decision tree model to the given training data.

        Parameters:
            X (array-like): The input features of the training data.
            y (array-like): The target values of the training data.
            max_depth (int): The maximum depth of the decision tree.

        Returns:
            None
        """
        self.tree = self.build_tree(X, y, max_depth=max_depth)

    def predict(self, sample):
            """
            Predicts the class label for a given sample using the decision tree.

            Parameters:
            sample (list): The feature values of the sample.

            Returns:
            int: The predicted class label.
            """
            current_node = self.tree
            while not current_node.is_leaf_node():
                if sample[current_node.feature] < current_node.threshold:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            return current_node.value







def load_data(file):
    """
    Load data from a file.

    Parameters:
    file (str): The path to the file.

    Returns:
    pandas.DataFrame: The loaded data.
    """
    data = pd.read_csv(file, encoding='utf-16', delimiter='\t')
    return data

def data_preprocessing_decision_tree(data):
    """
    Preprocesses the data for a decision tree model by categorizing the 'utility' column into labels.

    Args:
        data (pandas.DataFrame): The input data containing the 'utility' column.

    Returns:
        pandas.DataFrame: The preprocessed data with a new 'Label' column.

    """
    data['Label'] = pd.cut(data['utility'], bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf], labels=[1, 2, 3, 4, 5])
    return data



def cross_validation_decision_tree(X, y, k=5, max_depth=15):
    """
    Perform k-fold cross validation for a decision tree classifier.

    Parameters:
    - X (numpy.ndarray): Input features.
    - y (numpy.ndarray): Target labels.
    - k (int): Number of folds for cross validation (default: 5).
    - max_depth (int): Maximum depth of the decision tree (default: 15).

    Returns:
    - mean_accuracy (float): Mean accuracy across all folds.
    - variance (float): Variance of accuracies across all folds.
    """
    # Shuffle the data
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    # Split the data into k folds
    fold_size = X.shape[0] // k
    accuracies = []

    for fold in range(k):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_train = np.concatenate([X_shuffled[:start], X_shuffled[end:]])
        y_train = np.concatenate([y_shuffled[:start], y_shuffled[end:]])
        X_val = X_shuffled[start:end]
        y_val = y_shuffled[start:end]

        # Initialize and train the model
        tree = decision_tree()
        tree.fit(X_train, y_train, max_depth=max_depth)

        # Predict and calculate accuracy
        predictions = [tree.predict(x) for x in X_val]
        accuracy = np.sum(predictions == y_val) / len(y_val)
        accuracies.append(accuracy)

    
    variance = np.var(accuracies)
    return np.mean(accuracies), variance

def cross_validation_back_prop(X, y, k=5, hidden_size=15):
    """
    Perform k-fold cross validation using back propagation neural network.

    Parameters:
    X (numpy.ndarray): Input data.
    y (numpy.ndarray): Target data.
    k (int): Number of folds for cross validation. Default is 5.
    hidden_size (int): Number of hidden units in the neural network. Default is 15.

    Returns:
    tuple: Mean squared error score and variance of the scores.
    """
    # Shuffle the data
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    # Split the data into k folds
    fold_size = X.shape[0] // k
    mse_scores = []

    for fold in range(k):
        start, end = fold * fold_size, (fold + 1) * fold_size
        X_train = np.concatenate([X_shuffled[:start], X_shuffled[end:]])
        y_train = np.concatenate([y_shuffled[:start], y_shuffled[end:]])
        X_val = X_shuffled[start:end]
        y_val = y_shuffled[start:end]

        # Initialize and train the model
        model = BackProp(X_train.shape[1], hidden_size, 1)
        loss = model.train_neural_network(X_train, y_train, epochs=20000, learning_rate=0.05)

        # Predict and calculate mean squared error
        predictions = [model.predict(x) for x in X_val]
        mse = model.mean_squared_error(y_val, predictions)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.var(mse_scores), len(loss)





    






if  __name__ == "__main__":

    #read in data
    file = 'data.txt'
    data = load_data(file)
    X = data.drop('utility', axis=1).values.astype(np.float32)
    y = data['utility'].values.astype(np.float32).reshape(-1, 1)


    #Cross-validation_back_prop
    hidden_size_choices = [4, 8, 12]
    print("Cross-validation for back_prop")
    for hidden_size in hidden_size_choices:
        mse, varience, epochs = cross_validation_back_prop(X, y, k=5, hidden_size=hidden_size)
        #i use the RMSE because it is easeir to interpret
        print(f"Hidden Size: {hidden_size}, MSE: {mse:.3f}, Variance: {varience:.8f}, RMSE: {np.sqrt(mse):.3f}, epochs taken: {epochs}")


    #train decision tree model example
    decision_tree_data = data_preprocessing_decision_tree(data)
    y = decision_tree_data['Label'].values

    print("Cross-validation for decision_tree")
    # Cross-validation_decision_tree
    max_depth_choices = [2, 5, 10, 15, 20]
    for max_depth in max_depth_choices:
        accuracy, varience = cross_validation_decision_tree(X, y, k=5, max_depth=max_depth)
        print(f"Max Depth: {max_depth}, Accuracy: {accuracy:.3f}, Variance: {varience:.6f}")



    # model = BackProp(X.shape[1], 15, 1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # loss = model.train_neural_network(X_train, y_train, epochs=200, learning_rate=0.01)
    # y_pred  = model.predict(X_test)
    # mse = model.mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error for my model: {mse:.3f}")
    # #model.plot_loss(loss)
    # print(len(loss))


    # #use sklearn back propagation to compare the result, import mse

    # model = MLPRegressor(hidden_layer_sizes=(15,), max_iter=200, learning_rate_init=0.01, random_state=42)
    # model.fit(X_train, y_train.ravel())
    # y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error for sklearn model: {mse:.3f}")








    
