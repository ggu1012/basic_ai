import numpy as np
from matplotlib import pyplot as plt

def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)

class GaussianKernel():
    """
    Description:
         Filter the value with a Gaussian smoothing kernel with lambda value, and returns the filtered value.
    """
    def __init__(self, l):
        self.lamdba = l
    
    def __call__(self, value):
        """
        Args:
            value (numpy array) : input value
        Returns:
            value (numpy array) : filtered value
        """

        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        ############################


class KNN_Classifier():
    def __init__(self,n_neighbors=5,weights=None):

        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        """
        Description:
            Fit the k-nearest neighbors classifier from the training dataset.
    
        Args:
            X (numpy array): input data shape == (N, D)
            y (numpy array): label vector, shape == (N, ) 
    
        Returns:
        """
        
        self.X = X
        self.y = y
        
    def kneighbors(self, X):
        """
        Description:
            Find the K-neighbors of a point.
            Returns indices of and distances to the neighbors of each point.
    
        Args:
            X (numpy array): Input data, shape == (N, D)
            
        Returns:
            dist(numpy array) : Array representing the pairwise distances between points and neighbors , shape == (N, self.n_neighbors)
            idx(numpy array) : Indices of the nearest points, shape == (N, self.n_neighbors)
                
        """
        
        N = X.shape[0]

        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        ############################
        
        assert dist.shape == (N, self.n_neighbors)
        assert idx.shape == (N, self.n_neighbors)
        
        return dist, idx
        
    
    def make_weights(self, dist, weights):
        """
        Description:
            Make the weights from an array of distances and a parameter ``weights``.

        Args:
            dist (numpy array): The distances.
            weights : weighting method used, {'uniform', 'inverse distance' or a callable}

        Returns:
            (numpy array): array of the same shape as ``dist``
        """

        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        ############################

    def most_common_value(self, val, weights, axis=1):
        """
        Description:
            Returns an array of the most common values.

        Args:
            val (numpy array): 2-dim array of which to find the most common values.
            weights (numpy array): 2-dim array of the same shape as ``val``
            axis (int): Axis along which to operate
        Returns:
            (numpy array): Array of the most common values.
        """

        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        ############################

    def predict(self, X):
        """ 
        Description:
            Predict the class labels for the input data.
            When you implement KNN_Classifier.predict function, you should use KNN_Classifier.kneighbors, KNN_Classifier.make_weights, KNN_Classifier.most_common_value functions.

        Args:
            X (numpy array): Input data, shape == (N, D)

        Returns:
            pred (numpy array): Predicted target, shape == (N,)
        """

        ### CODE HERE ###
        raise NotImplementedError("Erase this line and write down your code.")
        ############################

def stack_accuracy_over_k(X_train, y_train, X_test, y_test, max_k=50, weights_list = ["uniform", "inverse distance", GaussianKernel(1000000)]):
    """ 
    Description:
        Stack accuracy over k.

    Args:
        X_train, X_test, y_train, y_test (numpy array)
        max_k (int): a maximum value of k
        weights_list (List[any]): a list of weighting method used
    Returns:
    """
    
    ### CODE HERE ###
    raise NotImplementedError("Erase this line and write down your code.")
    ############################
    
def knn_query(X_train, X_test, X_train_image, X_test_image, y_train, y_test, names, n_neighbors=5, n_queries=5):
    np.random.seed(42)
    my_clf = KNN_Classifier(n_neighbors=n_neighbors, weights="uniform")
    my_clf.fit(X_train, y_train)

    data = [(X_train, y_train, X_train_image), (X_test, y_test, X_test_image)]
    train = True
    for X, y, image in data:
        for i in range(n_queries):
            fig = plt.figure(figsize=(16, 6))
            rnd_indice = np.random.randint(low=X.shape[0], size=n_queries)
            nn_dist, nn_indice = my_clf.kneighbors(X)

            idx = rnd_indice[i]
            query = image[idx]
            name = names[y[idx]]
            prediction = my_clf.most_common_value(y_train[nn_indice[idx]], None, axis=0).astype(np.int8)
            prediction = names[prediction[0]]

            plt.subplot(1, n_neighbors + 1, 1)
            plt.imshow(query, cmap=plt.cm.bone)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.xlabel(f'Label: {name}\nPrediction: {prediction}')
            if i == 0:
                plt.title('query')

            for k in range(n_neighbors):
                nn_idx = nn_indice[idx, k]
                dist = nn_dist[idx, k]
                value = X_train_image[nn_idx]
                name = names[y_train[nn_idx]]
                
                plt.subplot(1, n_neighbors + 1, k + 2)
                plt.imshow(value, cmap=plt.cm.bone)
                plt.xticks([], [])
                plt.yticks([], [])
                plt.xlabel(f'Label: {name}\nDistance: {dist:0.2f}')
            plt.tight_layout()
            if i == 0:
                if train:
                    plt.suptitle(f'k nearest neighbors of queries from the training dataset', fontsize=30)
                    train = False
                else:
                    plt.suptitle(f'k nearest neighbors of queries from the test dataset', fontsize=30)
        
       




