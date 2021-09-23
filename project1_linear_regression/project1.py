from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    """
    Dataset class
    """
    
    def __init__(self, data_path):
        """
        Description:
            Load a csv file using 'load_data' method.
            Convert the dataframe to numpy array.
        
        Args:
            data_path (str): Path of the csv file.
                       
        Returns:
            
        """
        assert type(data_path)==str, 'The type of data_path has to be string'
        
        self.path = data_path
        self.features, self.data = self.load_data()
        self.min = None
        self.max = None        
                
    def load_data(self):
        """
        Description:
            Load the csv file.
            Print the head of data. 
        
        Args: 
            
            
        Returns:
            features (list): The names of the features.
            data (numpy array): Data.
        """
        dataframe = pd.read_csv(self.path)
        print(dataframe.head())
        features = list(dataframe)[:-1]
        data = dataframe.values
        return features, data
    
    def normalize(self):
        """
        Description:
            Do a min-max normalization
        
        Args:
        
        Returns:
            data (numpy array): Normalized data
            
        """
        self.min = self.data.min(axis=0)
        self.max = self.data.max(axis=0)
        data = (self.data - self.min) / (self.max - self.min)
        return data

    def parse_data(self, features):
        """
        Description:
            Parse the data using 'features'
            Do a normalize using 'normalize' method.
            Add a bias to the data.
        
        Args:
            features (list): The names of feature we use.
        
        Returns:
            X (numpy array): Input data
            y (numpy arary): Target (feature name: 'MEDV')
            
        """
        assert type(features)==list, 'The type of feature_names has to be list'
        assert all([isinstance(feature, str) for feature in features]), 'The element of features has to be string'
        
        data = self.normalize()
            
        indices = [self.features.index(feature) for feature in features]
        X = data[:, indices]
        y = data[:, -1]
        
        bias = np.ones([len(self.data), 1])
        X = np.concatenate((bias, X), axis=1)
        
        return X, y

    
class LinearRegressor:
    """
    Linear regressor class
    """
    def __init__(self, lr, tau, dim):
        """
        Description:
            Set the following attributes. 
                
                lr: learning rate
                tau: convergence tolerance
                dim: dimension of weight
                weight: regression coefficient
                loss_history: history of loss over the number of iterations.
        
        Args:
            lr (float): Learning rate.
            tau (float): Convergence condition.
            dim (int) : Dimension of weight.
            
        Returns:
            
        """

        self.lr = lr # float
        self.tau = tau # float
        self.dim = dim # int

        self.weight = np.zeros(dim)        
        self.weight[0] = 1 # Bias = 1

        self.loss_history = np.array([])
        
    
    def prediction(self, X):
        """
        Description: 
            Predict the target variable
            
        Args:
            X (numpy array): Input data
            
        Returns:
            pred (numpy array or float): Predicted target.
        """
        
        ### CODE HERE ###

        pred = np.dot(X, self.weight)

        #################
        
        return pred
    
    def compute_gradient(self, X, y):
        """
        Description:
            Calculate error and gradient.
        
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        
        Returns:
            error (numpy array or float): Error.
            gradient (numpy array): Gradient.
        """
        
        ### CODE HERE ###

        pred = self.prediction(X)
        N = y.shape[0]

        y_Hw = y - pred

        # RSS
        error = np.dot(np.transpose(y_Hw), y_Hw)
        # Average RSS
        error = error / N 

        gradient = -2 * np.dot(np.transpose(X), y_Hw) / N

        #################
        
        return error, gradient
    
    def LR_with_gradient_descent(self, X, y):
        """
        Description:
            Do a gradient descent.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            
        Returns:
            
        """
        
        ### CODE HERE ###

        ### Initialization 

        threshold = self.tau
        lr = self.lr

        error, gradient = self.compute_gradient(X,y)
        infNorm = lr * np.max(np.abs(gradient))
        self.loss_history = np.concatenate((self.loss_history, [error]))

        ### Gradient descent
        while(infNorm >= threshold):    
            self.weight = self.weight - lr * gradient

            error, gradient = self.compute_gradient(X,y)            
            infNorm = lr * np.max(np.abs(gradient))
            self.loss_history = np.concatenate((self.loss_history, [error]))

        
        
    def plot_loss_history(self):
        """
        Description:
            Plot the history of the loss.
        
        Args:
        
        Returns:
        
        """        
        
        ### CODE HERE ###

        plt.plot(self.loss_history)
        plt.xlabel('iterations')
        plt.ylabel('Average loss')
        plt.title('Average loss over # of iterations')

        #################
        
        
    def calculate_house_price(self, input_features, min_value, max_value):
        """
        Description:
            Predict the house price.
        
        Args:
            input_features (numpy array): Input data.
            min_value (float): The minimum value of target.
            max_value (float): The maximum value of target.
        
        Returns:
            price (float): Predicted price.
            
        
        """
        
        ### CODE HERE ###
        
        featureSize = input_features.shape[0]
        normalizedInputFeatures = np.ones(featureSize + 1) # +1 for bias
        for i in range(featureSize):
            normalizedInputFeatures[i+1] = (input_features[i] - min_value[i]) / (max_value[i] - min_value[i])
        
        normalizedPrice = np.dot(normalizedInputFeatures, self.weight)
        price = normalizedPrice * (max_value[featureSize] - min_value[featureSize]) + min_value[featureSize]
        
        #################
        
        return price

    
def LR_with_closed_form(X, y):
    
    """
    Description:
        Do linear regression without iteration.
            
    Args:
        X (numpy array): Input data.
        y (numpy array or float): Target data.
            
    Returns:
            
    """
    ### CODE HERE ###
    
    HT = np.transpose(X)
    HTH = np.dot(HT, X)
    weight = np.dot(np.dot(np.linalg.inv(HTH), HT), y) # inv(HTH)*HT*y


    #################
    return weight
    

def visualize(weight, X, y, features):
    """
    Description:
        Plot the data distribution and lines using weight vector.
        
    Args:
        weight (numpy array): Weight vector.
        X (numpy array): Input data.
        y (numpy array): Target data.
        
    Returns:
    
    """
    plt.figure(figsize=(30, 30))
    for i in range(1, X.shape[1]):
        pred_space = np.linspace(min(X[:, i]), max(X[:, i]), len(X[:, 0]))
        plt.subplot(4, 4, i)
        plt.scatter(X[:, i], y)
        plt.plot(pred_space, pred_space * weight[i] + weight[0], color='red', linewidth=4.0)
        plt.xlabel(features[i - 1])
        plt.ylabel('MEDV')
    plt.show()

