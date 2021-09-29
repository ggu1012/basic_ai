from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
class LinearRegressor:
    """
    LinearRegressor class with 'coordinate descent'.
    """
    def __init__(self, tau, dim):
        """
        Description:
            Set the attributes. 
                
                tau: convergence tolerance.
                dim: dimension of weight.
                weight: regression coefficient.
                loss_history: history of RSS loss over the number of iterations.
        
        Args:
            tau (float): Convergence condition.
            dim (int) : Dimension of weight.
            
        Returns:
            
        """
        
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
    
    def initialize_weight(self):
        """
        Description: 
            Initialize the weight randomly.
            Use the normal distribution.
            
        Args:
            
        Returns:
            
        """
        np.random.seed(0)
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
    
    
    def prediction(self, X):
        """
        Description: 
            Predict the target variable.
            
        Args:
            X (numpy array): Input data
            
        Returns:
            pred (numpy array or float): Predicted target.
        """
        
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        return pred
    
    def compute_residual(self, X, y):
        """
        Description:
            Calculate residual between prediction and target.
        
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        
        Returns:
            residual (numpy array or float): residual.
        """
        
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        return residual
    
    def LR_with_coordinate_descent(self, X, y):
        """
        Description:
            Do a coordinate descent.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            
        Returns:
            
        """
        
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        
        
    def plot_loss_history(self):
        """
        Description:
            Plot the history of the RSS loss.
        
        Args:
        
        Returns:
        
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        
        
class RidgeRegressor(LinearRegressor):
    """
    RidgeRegressor class. 
    You should inherit the LinearRegressor as base class.
    """
    def __init__(self, tau, dim, lambda_):
        """
        Description:
            Set the attributes. You can use super().
                
                tau: convergence tolerance.
                dim: dimension of weight.
                weight: regression coefficient.
                loss_history: history of RSS loss over the number of iterations.
                lambda_ : hyperparameter for regularization.
        
        Args:
            tau (float): Convergence condition.
            dim (int): Dimension of weight.
            lambda_ (float or int): Hyperparameter for regularization.
            
        Returns:
            
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        
    def LR_with_coordinate_descent(self, X, y):
        """
        Description:
            Do a coordinate descent. Do not penalize the intercept term.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            
        Returns:
            
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
    
    
class LassoRegressor(LinearRegressor):
    """
    LassoRegressor class. 
    You should inherit the LinearRegressor as base class.
    """
    def __init__(self, tau, dim, lambda_):
        """
        Description:
            Set the attributes. You can use super().
                
                tau: convergence tolerance.
                dim: dimension of weight.
                weight: regression coefficient.
                loss_history: history of RSS loss over the number of iterations.
                lambda_: hyperparameter for regularization.
                
        Args:
            tau (float): Convergence condition.
            dim (int) : Dimension of weight.
            lambda_ (float or int): Hyperparameter for regularization.
            
        Returns:
            
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
    
    def LR_with_coordinate_descent(self, X, y):
        """
        Description:
            Do a coordinate descent. Do not penalize the intercept term.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            
        Returns:
            
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################


def stack_weight_over_lambda(X, y, model_type, tau, dim, lambda_list):
    """
        Description:
            Calcualte the regression coefficients over lambdas and stack the results.
            
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            mdoel_type (str): Type of model
            dim (int): Dimension of weight.
            lambda_list (list): List of lambdas.
            
        Returns:
            stacked_weight (numpy array): Weight stacked over lambda.
            
    """
    assert model_type in ['Lasso', 'Ridge'], f"model_type must be 'Ridge' or 'Lasso' but were given {model_type}"
    stacked_weight = np.zeros([len(lambda_list), X.shape[1]])
    ### CODE HERE ###
    raise NotImplementedError("Erase this line and write down your code.")
    #################
    return stacked_weight


def get_number_of_non_zero(weights):
    """
        Description:
            Find the number of non-zero weight in regression coefficients over lambdas.
            
        Args:
            weights (numpy array): Regression coefficients over lambdas.
            
        Returns:
            num_non_zero (list): Number of non-zero coefficients over lambdas.
    """
    num_non_zero = []
    ### CODE HERE ###
    raise NotImplementedError("Erase this line and write down your code.")
    #################
    return num_non_zero


def compute_errors(X, y, lambda_list, weights):
    """
        Description:
             Calcualte the RSS error between predictions and target values using 
             the output of stack_weight_over_lambda.
             
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
            lambda_list (list): List of lambdas.
            weights (numpy array): Stacked weights.
            
        Returns:
            rss_errors (list): List of RSS errors calculated over lambdas.
    """
    assert len(lambda_list) == len(weights)
    rss_errors = []
    ### CODE HERE ###
    raise NotImplementedError("Erase this line and write down your code.")
    #################
    return rss_errors

