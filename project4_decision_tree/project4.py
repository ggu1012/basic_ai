import numpy as np
from numpy.lib.function_base import select
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

    X = df.drop(["label"], axis=1, inplace=False).to_numpy()
    y = df["label"].to_numpy()

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

        """
        Node elements        
        
        idx : current node number
        feature : splitting feature
        threshold : threshold value        
        is_leaf : 0 as default, it would be set as 1 in leaf_node()       
        
        """

        self.node_stack = []
        self.node_info = [[] for row in range(self.max_depth + 2)]
        self.depth = 0
        self.node_idx = 0

        full_indice = np.arange(self.X.shape[0])

        # Initial node 
        node = dict()
        node['idx'] = self.node_idx
        node['prediction'] = self.node_prediction(full_indice)
        node['is_leaf'] = 0

        right, left, node['feature'], node['threshold'] = self.best_split(np.arange(self.X.shape[0]))

        node['impurity_before'] = self.compute_gini_impurity(full_indice, [])
        node['impurity_after'] = self.compute_gini_impurity(left, right)        
        
        self.node_stack.append(right)
        self.node_stack.append(left)        
        self.node_info[self.depth].append(node)

        self.depth += 1        

        while self.node_stack:            
            self.node_idx += 1            
            to_be_splitted = self.node_stack.pop()     

            right, left, selected_feature, optimal_threshold = self.best_split(to_be_splitted)        
            impurity_before = self.compute_gini_impurity(to_be_splitted, [])
            impurity_after = self.compute_gini_impurity(left, right)               

            # Early stopping condition
            if self.depth >= self.max_depth \
                or np.size(to_be_splitted) <= self.min_splits \
                or impurity_after >= impurity_before:
                self.node_info[self.depth].append(self.leaf_node(to_be_splitted)) 

                for n in reversed(range(self.depth + 1)):
                    if len(self.node_info[n]) % 2 != 0:
                        break                    
                    self.depth = self.depth - 1                                

                continue     



            node = dict()
            node['idx'] = self.node_idx
            node['prediction'] = self.node_prediction(to_be_splitted)
            node['is_leaf'] = 0    
            node['feature'] = selected_feature
            node['threshold'] = optimal_threshold
            node['impurity_before'] = impurity_before
            node['impurity_after'] = impurity_after          
            self.node_info[self.depth].append(node)

            self.node_stack.append(right)
            self.node_stack.append(left)  

            self.depth = self.max_depth if self.depth >= self.max_depth \
                else self.depth + 1      
      
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

        # Computes node impurity        

        if np.size(left_index) == 0:
            diabetes_size = np.count_nonzero(self.y[right_index])
            prob_diabetes = diabetes_size / np.size(right_index)
            prob_normal = 1 - prob_diabetes  

            return 1 - (prob_normal**2 + prob_diabetes**2)

        elif np.size(right_index) == 0:
            diabetes_size = np.count_nonzero(self.y[left_index])
            prob_diabetes = diabetes_size / np.size(left_index)
            prob_normal = 1 - prob_diabetes

            return 1 - (prob_normal**2 + prob_diabetes**2)

        # Computes weighted impurity
        else:    
            gini_left = self.compute_gini_impurity(left_index, [])
            gini_right = self.compute_gini_impurity([], right_index)            

            N_left = left_index.size
            N_right = right_index.size
            N = N_left + N_right
            
            weighted_impurity = (N_left / N) * gini_left + (N_right / N) * gini_right

            return weighted_impurity

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

        node = dict()
        node['idx'] = self.node_idx   
        node["prediction"] = self.node_prediction(index)
        node["impurity"] = self.compute_gini_impurity(index,[]) 
        node["is_leaf"] = 1

        return node
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

        normal_indice = np.argwhere(self.y[index] == 0).flatten()
        normal = index[normal_indice]
        diabetes = np.delete(index, normal_indice)

        if np.size(normal) >= np.size(diabetes):
            return 0
        else:
            return 1

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

        feature_num = self.X.shape[1]        
        threshold_list = []
        
        for feature in range(feature_num):

            # Sort X[index,j] in ascending order and pick unique values
            ascending_order = np.unique(np.sort(self.X[index, feature]))
            

            for idx in range(ascending_order.size):               

                #####
                threshold = ascending_order[idx]
                indice_idx_number = np.argwhere(self.X[index, feature] <= threshold).flatten()
                
                left = index[indice_idx_number]
                right = np.delete(index, indice_idx_number)                

                weighted_impurity = self.compute_gini_impurity(left, right)

                threshold_dict = dict() # {feature, impurity, threshold value}
                threshold_dict['feature'] = feature
                threshold_dict['impurity'] = weighted_impurity
                threshold_dict['threshold'] = threshold

                threshold_list.append(threshold_dict)


                ####
                if idx != ascending_order.size - 1:
                    threshold = (ascending_order[idx] + ascending_order[idx + 1]) / 2
                    indice_idx_number = np.argwhere(self.X[index, feature] <= threshold).flatten()
                    left = index[indice_idx_number]
                    right = np.delete(index, indice_idx_number)

                    weighted_impurity = self.compute_gini_impurity(left, right)

                    threshold_dict = dict() # {feature, impurity, threshold value}                    
                    threshold_dict['feature'] = feature
                    threshold_dict['impurity'] = weighted_impurity
                    threshold_dict['threshold'] = threshold

                    threshold_list.append(threshold_dict)
                    

        min_item = min(threshold_list, key=lambda x:x['impurity'])
        selected_feature = min_item['feature']
        optimal_threshold = min_item['threshold']
        
        indice_idx_number = np.argwhere(self.X[index, selected_feature] <= optimal_threshold).flatten()
        left_indice = index[indice_idx_number]
        right_indice = np.delete(index, indice_idx_number)  

    
        return right_indice, left_indice, selected_feature, optimal_threshold

      
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

        pred = np.zeros(X.shape[0])

        

        for sample_idx, sample in enumerate(X):

            column = 0
            for depth in range(self.max_depth + 2):
                selected_node = self.node_info[depth][column]

                if selected_node['is_leaf'] == 1:                    
                    pred[sample_idx] = selected_node['prediction']
                    break
                    # print("END! prediction = %d" %(pred[sample_idx]))
                    

                
                selected_feature = selected_node['feature']
                threshold = selected_node['threshold']
                node_idx = selected_node['idx']

                for num, node in enumerate(self.node_info[depth+1]):
                    if node['idx'] == node_idx + 1 :                        
                        break

                if sample[selected_feature] <= threshold:                    
                    column = num
                    # print("%.4f <= %.4f : goes to node %d" %(sample[selected_feature], threshold,\
                    # self.node_info[depth+1][column]['idx']))
                else:
                    column = num + 1         
                    # print("%.4f > %.4f : goes to node %d" %(sample[selected_feature], threshold,\
                    # self.node_info[depth+1][column]['idx']))


        return pred
        


        #################
    
    def traverse(self):
        """ 
        Description:
            Traverse through the tree in Breadth First Search fashion to compute various properties.
        
        Args:
        
        Returns:
        """
        ### CODE HERE ###

        for depth in range(self.max_depth + 1):            
            for node in self.node_info[depth]:
                if node['is_leaf'] == 0:

                    right_node_idx = 0
                    for next_node in self.node_info[depth+1]:
                        right_node_idx += 1
                        if next_node['idx'] == node['idx'] + 1 :                        
                            break
                        
                    print("\t"*depth, end='')
                    print("node=%d is a split node: go to left node %d if self.X[:, %d] <= %.4f else to right node %d: Impurity: %.4f, Improvement: %.4f, Prediction -> %d" \
                         %(node['idx'], node['idx'] + 1, node['feature'], node['threshold'], \
                             self.node_info[depth+1][right_node_idx]['idx'], node['impurity_before'], \
                                 node['impurity_before']-node['impurity_after'], node['prediction']))

                else:
                    print("\t"*depth, end='')
                    print("node=%d is a leaf node: Impurity %.4f, Prediction -> %d" %(node['idx'], node['impurity'], node['prediction']))


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

    accuracy_train = []
    accuracy_test = []
    final_depth = []
    number_of_nodes = []

    depth_range = range(1, 16)
    
    for max_depth in depth_range:

        ########### Training set

        tree = DecisionTree(max_depth, min_splits)
        tree.fit(X_train, y_train)

        y_pred = tree.predict(X_train)
        accuracy_train.append(accuracy(y_train, y_pred))

        max_dp = 0
        num_of_nodes = 0
        for n in range(max_depth+1):
            if len(tree.node_info[n]) == 0:
                break
            max_dp += 1
            num_of_nodes += len(tree.node_info[n])

        final_depth.append(max_dp)
        number_of_nodes.append(num_of_nodes)

        ###########

        ########### Test set
        
        y_pred = tree.predict(X_test)
        accuracy_test.append(accuracy(y_test, y_pred))

        ###########


    plt.figure(figsize=[20,4])

    plt.subplot(1,3,1)
    plt.plot(depth_range, accuracy_train, label='train accuracy')
    plt.plot(depth_range, accuracy_test, label='test accuracy')
    plt.legend()
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')

    plt.subplot(1,3,2)
    plt.plot(depth_range, final_depth)
    plt.xlabel('max_depth')
    plt.ylabel('Depth')

    plt.subplot(1,3,3)
    plt.plot(depth_range, number_of_nodes)
    plt.xlabel('max_depth')
    plt.ylabel('Number of nodes')



        









    #################


    