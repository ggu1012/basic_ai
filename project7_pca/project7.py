from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt


class PCA:
    """PCA (Principal Components Analysis) class."""
    def __init__(self, num_components):
        """
        Descriptions:
            Constructor
        
        Args:
            num_components: (int) number of component to keep during PCA.  
        
        Returns:
            
        """
        self.num_components = num_components
        
        assert isinstance(self.num_components, int)

    
    def find_principal_components(self, X):
        """
        Descriptions:
            Find the principal components. The number of components is num_components.
            Set the class attribute, X_mean which represent the mean of training samples.
            
        Args:
            X : (numpy array, shape is (number of samples, dimension of feature)) training samples
                  
        Returns:
            
            
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        
        assert self.eigenbasis.shape == (self.num_components, X.shape[1])
                                 
        
    def reduce_dimensionality(self, samples):
        """
        Descriptions:
            Reduce the dimensionality of data using the principal components. Before project the samples onto eigenspace,
            you should standardize the samples.
            
        Args:
            samples: (numpy array, shape is (number of samples, dimension of features))
                
        Returns:
            data_reduced: (numpy array, shape is (number of samples, num_components).) Data representation with only
                          num_components of the basis vectors.
                
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################        
        assert data_reduced.shape == (samples.shape[0], self.num_components)

        return data_reduced
    
    
    def reconstruct_original_sample(self, sample_decomposed):
        """
        Descriptions:
            Normalize the training samples.
            
        Args:
            sample_decomposed: (numpy array, shape is (num_components, ).) Sample which decomposed using principal components
            keeped from PCA.
                
        Returns:
            representations_onto_eigenbasis: (numpy array, shape is (num_components, dimension of original feature).) 
            New feature reperesntation using eigenbasis which keeped from PCA.
            
            sample_recovered: (numpy array, shape is (dimension of original feature).) 
            Sample which recovered with linearly combined eigenbasis.
                
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        
        return representations_onto_eigenbasis, sample_recovered
    
    
class FaceRecognizer(PCA):
    """FaceRecognizer class."""
    def __init__(self, num_components, X, y):
        """
        Descriptions:
            Constructor. Inherit the PCA class.
        
        Args:
            num_components: (int) number of component to keep during PCA.  
            X : (numpy array, shape is (number of samples, dimension of feature)) training samples.
            y : (numpy array, shape is (number of samples, )) lables of corresponding samples.
        
        Returns:
        """
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        
    
    def generate_database(self):
        """
        Descriptions:
            Generate database using eigenface.
        
        Args:
        
        Returns:
        """
        
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        
    
    def find_nearest_neighbor(self, X):
        """
        Descriptions:
            Find the nearest sample in the database.
        
        Args:
            X : (numpy array, shape is (number of samples, dimension of feature)) Query samples.
        
        Returns:
            pred: (numpy array, shape is (number of queries, )) Predictions of each query sample.
            distance: (numpy array, shape is (number of queries, 1)) Distances between query samples and corresponding DB.
            db_indices: (numpy array, shape is (number of queries, )) Indices of nearest samples in DB.
        """
        
        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        #################
        
        return pred, distances, db_indices  
    

        
