import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

INFINITY = np.inf
EPSILON = np.finfo('double').eps


def load_data(df):
    """Return X, y and features.
    
    Args:
        df: pandas.DataFrame object.
    
    Returns:
        Tuple of (X, y)
        X (ndarray): include the columns of the features, shape == (N, D)
        y (ndarray): label vector, shape == (N, )
    """
    
    N = df.shape[0] # the number of samples
    D = df.shape[1] - 1 # the number of features, excluding a label
    
    ### CODE HERE ###
    raise NotImplementedError("Erase this line and write down your code.")
    #################

    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape == (N, D) and y.shape == (N, ), f'{(X.shape, y.shape)}'
    
    return X, y


def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)

class DecisionTree(object):
    def __init__(self, max_depth, min_splits):
        self.max_depth = max_depth
        self.min_splits = min_splits

    def fit(self, X, y):
        """
        Description:
            Return X, y and features.
    
        Args:
            X (numpy array): Input data shape == (N, D)
            y (numpy array): label vector, shape == (N, ) 
    
        Returns:
        """
        
        self.X = X
        self.y = y
        
        self.build()
    
    def build(self):
        """
        Description:
            Build a binary tree in Depth First Search fashion
                - Make a internal node using split funtion or a leaf node using leaf_node function
                - Use a stack to build a binary tree
                - Consider stop condition & early stop condition
        Args:
        Returns:
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################

    def compute_gini_impurity(self, left_index, right_index):
        """
        Description:
            Compute the gini impurity for the indice 
                - if one of arguments is empty array, it computes node impurity
                - else, it computes weighted impurity of both sub-nodes of that split.

        Args:
            left_index (numpy array): indice of data of left sub-nodes  
            right_index (numpy array): indice of data of right sub-nodes

        Returns:
            gini_score (float) : gini impurity
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################

    def leaf_node(self, index):
        """ 
        Description:
            Make a leaf node(dictionary)

        Args:
            index (numpy array): indice of data of a leaf node

        Returns:
            leaf_node (dict) : leaf node
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################

    def node_prediction(self, index):
        """ 
        Description:
            Make a prediction(label) as the most common class

        Args:
            index (numpy array): indice of data of a node

        Returns:
            prediction (int) : a prediction(label) of that node
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
    
    def best_split(self, index):
        """ 
        Description:
            Find the best split information using the gini score and return a node

        Args:
            index (numpy array): indice of data of a node

        Returns:
            node (dict) : a split node that include the best split information(e.g., feature, threshold, etc.)
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################

    def predict(self, X):
        """ 
        Description:
            Determine the class of unseen sample X by traversing through the tree.

        Args:
            X (numpy array): Input data, shape == (N, D)

        Returns:
            pred (numpy array): Predicted target, shape == (N,)
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
    
    def traverse(self):
        """ 
        Description:
            Traverse through the tree in Breadth First Search fashion to compute various properties.
        
        Args:
        
        Returns:
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
            

def plot_graph(X_train, X_test, y_train, y_test, min_splits = 2):
    """
    Description:
        Plot the depth, the number of nodes and the classification accuracy on training samples and test samples by varying maximum depth levels of a decision tree from 1 to 15.
    Args:
        X_train, X_test, y_train, y_test (numpy array)

    Returns:
    """
    ### CODE HERE ###
    raise NotImplementedError("Erase this line and write down your code.")
    #################


    