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

        self.tau = tau
        self.dim = dim
        self.weight = []
        self.loss_history = []

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

        self.weight = np.random.normal(0, 1, self.dim)

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

        pred = np.dot(X, self.weight)

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

        pred = self.prediction(X)
        residual = np.sum((y - pred)**2)

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

        self.initialize_weight()
        weightDiff = np.zeros(self.dim)  

        iter=1
        while(1):            
            iter=iter+1           

            ### for 1...w_d coordinate descent
            for j in range(self.dim):

                featureDeleted = np.delete(X, j, axis = 1)
                weightDeleted = np.delete(self.weight, j)
                predDeleted = np.dot(featureDeleted, weightDeleted)

                diffDeleted = y - predDeleted
                h_j = X[:,j]
                rho_j = np.dot(h_j.T, diffDeleted)

                z_j = np.sum(h_j**2)
                w_t = rho_j / z_j
                weightDiff[j] = np.abs(w_t - self.weight[j])
                self.weight[j] = w_t
            
            residual = self.compute_residual(X, y)
            self.loss_history = np.concatenate((self.loss_history, [residual]))
 
            if(np.max(weightDiff) < self.tau):              
                break

        #################
        
        
    def plot_loss_history(self):
        """
        Description:
            Plot the history of the RSS loss.
        
        Args:
        
        Returns:
        
        """
        ### CODE HERE ###
        plt.plot(self.loss_history)
        plt.xlabel('iterations')
        plt.ylabel('RSS loss')
        plt.title('RSS loss over # of iterations')
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
        super().__init__(tau, dim)
        self.lambda_ = lambda_
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
        self.initialize_weight()
        weightDiff = np.zeros(self.dim)  

        iter=1
        while(1):            
            iter=iter+1            

            ### for 1...w_d coordinate descent
            for j in range(self.dim):

                featureDeleted = np.delete(X, j, axis = 1)
                weightDeleted = np.delete(self.weight, j)
                predDeleted = np.dot(featureDeleted, weightDeleted)

                diffDeleted = y - predDeleted
                h_j = X[:,j]
                rho_j = np.dot(h_j.T, diffDeleted)
                z_j = np.sum(h_j**2)

                # Do not penalize intercept
                if j==0:
                    w_t = rho_j / z_j
     
                else:                                   
                    w_t = rho_j / (z_j + self.lambda_)
                
                weightDiff[j] = np.abs(w_t - self.weight[j])
                self.weight[j] = w_t 
            
            residual = self.compute_residual(X, y)
            self.loss_history = np.concatenate((self.loss_history, [residual]))
 
            if(np.max(weightDiff) < self.tau):           
                break

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
        super().__init__(tau, dim)
        self.lambda_ = lambda_
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

        self.initialize_weight()
        weightDiff = np.zeros(self.dim)  

        iter=1
        while(1):            
            iter=iter+1            

            ### for 1...w_d coordinate descent
            for j in range(self.dim):

                featureDeleted = np.delete(X, j, axis = 1)
                weightDeleted = np.delete(self.weight, j)
                predDeleted = np.dot(featureDeleted, weightDeleted)

                diffDeleted = y - predDeleted
                h_j = X[:,j]
                rho_j = np.dot(h_j.T, diffDeleted)
                z_j = np.sum(h_j**2)

                # Do not penalize intercept
                if j==0:
                    w_t = rho_j / z_j
     
                else: 
                    halfLambda = self.lambda_ / 2
                    if rho_j < -halfLambda:                                  
                        w_t = (rho_j + halfLambda) / z_j                    
                    elif (-halfLambda) <= rho_j and rho_j <= (halfLambda):
                        w_t = 0
                    else:
                        w_t = (rho_j - halfLambda) / z_j
                
                weightDiff[j] = np.abs(w_t - self.weight[j])
                self.weight[j] = w_t 
            
            residual = self.compute_residual(X, y)
            self.loss_history = np.concatenate((self.loss_history, [residual]))
 
            if(np.max(weightDiff) < self.tau):            
                break
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
    for idx, lambda_ in enumerate(lambda_list):
        print("%d" %(lambda_), end=" ")
        if model_type == 'Ridge':
            ridge = RidgeRegressor(tau=tau, dim=dim, lambda_=lambda_)
            ridge.LR_with_coordinate_descent(X, y)
            stacked_weight[idx] = ridge.weight
        else:
            lasso = LassoRegressor(tau=tau, dim=dim, lambda_=lambda_)
            lasso.LR_with_coordinate_descent(X, y)
            stacked_weight[idx] = lasso.weight
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
    for lambda_ in range(weights.shape[0]):
        num_non_zero = np.concatenate((num_non_zero, [np.count_nonzero(weights[lambda_])]))
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

    for lambda_ in range(weights.shape[0]):
        pred = np.dot(X, weights[lambda_])
        rss = np.sum((y - pred) ** 2)
        rss_errors = np.concatenate((rss_errors, [rss]))

    #################
    return rss_errors

