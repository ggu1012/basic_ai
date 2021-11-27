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

        covariance = np.cov(X.T)
        eigval, eigvec = np.linalg.eig(covariance)

        arg = np.argsort(eigval)[-self.num_components:][::-1]
        self.eigenbasis = eigvec[:,arg].T.real

        self.X_mean = np.average(X, axis=0)
        self.X_std = np.std(X, axis=0).reshape(1,-1)

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

        sample_mean = np.average(samples, axis=0)
        sample_std = np.std(samples, axis=0).reshape(1,-1)
        standardized_samples = (samples - sample_mean) / sample_std
        data_reduced = standardized_samples.dot(self.eigenbasis.T)

        self.sample_mean = sample_mean
        self.sample_std = sample_std

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

        representations_onto_eigenbasis = np.multiply(sample_decomposed.reshape(-1,1), self.eigenbasis)
        sample_recovered = sample_decomposed.dot(self.eigenbasis)
        sample_recovered = self.sample_std * sample_recovered + self.sample_mean

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

        super().__init__(num_components)
        self.X = X
        self.y = y

        #################
        
    
    def generate_database(self):
        """
        Descriptions:
            Generate database using eigenface.
        
        Args:
        
        Returns:
        """
        
        ### CODE HERE ###
        
        self.find_principal_components(self.X)
        self.database = self.reduce_dimensionality(self.X)

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

        sample_reduced = self.reduce_dimensionality(X)
        db_indices = np.empty((0,0), int)
        distances = np.empty((0,0), float)

        for sample in sample_reduced:
            distance_btwn_point = np.array([np.sqrt(((sample - self.database[i])**2).sum()) for i in range(self.database.shape[0])])
            nearest_idx = np.argmin(distance_btwn_point)
            db_indices = np.append(db_indices, nearest_idx)
            distances = np.append(distances, distance_btwn_point[nearest_idx])

        pred = self.y[db_indices]
        distances = distances.reshape((-1, 1))

        #################
        
        return pred, distances, db_indices  
    

        
