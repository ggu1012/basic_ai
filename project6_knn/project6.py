import numpy as np
from matplotlib import pyplot as plt

def accuracy(y_true, y_pred):
    return np.average(y_true == y_pred)


class GaussianKernel:
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

        return np.exp(-(value**2) / self.lamdba)

        ############################


class KNN_Classifier:
    def __init__(self, n_neighbors=5, weights=None):

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

        dist = np.empty((0,self.n_neighbors), int)
        idx = np.empty((0,self.n_neighbors), int)

        # distance_btwn_point = pairwise_distances(X, self.X)

        # idx = np.array([np.argsort(row)[0:self.n_neighbors] for row in distance_btwn_point])
        # dist = np.array([[distance_btwn_point[j, i] for i in idx[j]] for j in range(len(idx))])

        for point_idx in range(N):
            
            distance_btwn_point = np.array([np.sqrt(((X[point_idx] - self.X[i])**2).sum()) for i in range(self.X.shape[0])])
            
            short_idx = np.argsort(distance_btwn_point)[0:self.n_neighbors]
            short_distance = [distance_btwn_point[i] for i in short_idx]

            idx = np.vstack((idx, short_idx))
            dist = np.vstack((dist, short_distance))

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

        if callable(weights) == True: # Gaussian kernel            
            return weights(dist)

        elif weights == 'uniform':
            return np.ones(dist.shape)

        elif weights == 'inverse distance':
            return np.array([[(1 / d) if d != 0 else 1 for d in one] for one in dist])

        else:
            raise NotImplementedError("Error: Such kernel type is unsupported.")

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

        if weights is None:
            weights = np.ones(val.shape[0])

        y_values = np.unique(val)

        cumulated_weights = [sum([weights[i] for i in np.argwhere(val==y_val).flatten()]) for y_val in y_values]
        sorted_cumulated_weights_args = np.argsort(cumulated_weights)[::-1]

        return y_values[sorted_cumulated_weights_args]


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

        pred = []
        dist, idx = self.kneighbors(X)

        kernel_weight = self.make_weights(dist, self.weights)

        for data_idx in range(X.shape[0]):

            indices = idx[data_idx, :]
            yval = self.y[indices]

            pred_one_value = self.most_common_value(yval, kernel_weight[data_idx], axis=0)
            pred.append(pred_one_value[0])

        return pred

        ############################


def stack_accuracy_over_k(
    X_train,
    y_train,
    X_test,
    y_test,
    max_k=50,
    weights_list=["uniform", "inverse distance", GaussianKernel(1000000)],
):
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

    

    plt.figure(figsize=(20,4))
    subplot_num = np.size(weights_list)

    weights_type_string = ["uniform", "inverse distance", "gaussian"]
    

    for sub, weight_type in enumerate(weights_list):

        train_acc = []
        test_acc = []

        xaxis = range(1, max_k+1)        
        for k in xaxis: # 1 ~ 50
            my_clf = KNN_Classifier(n_neighbors=k, weights=weight_type)
            my_clf.fit(X_train, y_train)

            train_pred = my_clf.predict(X_train)
            train_acc.append(accuracy(y_train, train_pred))

            test_pred = my_clf.predict(X_test)
            test_acc.append(accuracy(y_test, test_pred))   
        
        plt.subplot(1, subplot_num, sub+1)
        plt.plot(xaxis, train_acc, label='train accuracy')
        plt.plot(xaxis, test_acc, label='test accuracy')
        plt.legend()
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over k with %s kernel' %(weights_type_string[sub]))
    
    plt.show()




    ############################


def knn_query(
    X_train,
    X_test,
    X_train_image,
    X_test_image,
    y_train,
    y_test,
    names,
    n_neighbors=5,
    n_queries=5,
):
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
            prediction = my_clf.most_common_value(
                y_train[nn_indice[idx]], None, axis=0
            ).astype(np.int8)
            prediction = names[prediction[0]]

            plt.subplot(1, n_neighbors + 1, 1)
            plt.imshow(query, cmap=plt.cm.bone)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.xlabel(f"Label: {name}\nPrediction: {prediction}")
            if i == 0:
                plt.title("query")

            for k in range(n_neighbors):
                nn_idx = nn_indice[idx, k]
                dist = nn_dist[idx, k]
                value = X_train_image[nn_idx]
                name = names[y_train[nn_idx]]

                plt.subplot(1, n_neighbors + 1, k + 2)
                plt.imshow(value, cmap=plt.cm.bone)
                plt.xticks([], [])
                plt.yticks([], [])
                plt.xlabel(f"Label: {name}\nDistance: {dist:0.2f}")
            plt.tight_layout()
            if i == 0:
                if train:
                    plt.suptitle(
                        f"k nearest neighbors of queries from the training dataset",
                        fontsize=30,
                    )
                    train = False
                else:
                    plt.suptitle(
                        f"k nearest neighbors of queries from the test dataset",
                        fontsize=30,
                    )

