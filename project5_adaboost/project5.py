import numpy as np
from numpy.random.mtrand import sample
import pandas as pd

from copy import deepcopy


def sign(x):
    return 2 * (x >= 0) - 1


class DecisionStump:
    """DecsionStump class"""

    def __init__(self):
        """
        Description:
            Set the attributes. 
                
                selected_feature (numpy.int): Selected feature for classification. 
                threshold: (numpy.float) Picked threhsold.
                left_prediction: (numpy.int) Prediction of the left node.
                right_prediction: (numpy.int) prediction of the right node.
        
        Args:
            
        Returns:
            
        """
        self.selected_feature = None
        self.threshold = None
        self.left_prediction = None
        self.right_prediction = None

    def fit(self, X, y):
        self.build_stump(X, y)

    def build_stump(self, X, y):
        """
        Description:
            Build the decision stump. Find the feature and threshold. And set the predictions of each node. 
        
        Args:
            X: (N, D) numpy array. Training samples.
            y: (N, ) numpy array. Target variable, has the values of 1 or -1.
                where N is the number of samples and D is the feature dimension.
            
        Returns:
            
        """
        ### CODE HERE ###

        self.select_feature_split(X, y)

        #################

    def select_feature_split(self, X, y):
        """
        Description:
            Find the best feature split. After find the best feature and threshold,
            set the attributes (selected_feature and threshold).
        
        Args:
            X: (N, D) numpy array. Training samples.
            y: (N, ) numpy array. Target variable, has the values of 1 or -1.
                where N is the number of samples and D is the feature dimension.
            
        Returns:
            
        """
        ### CODE HERE ###

        decision_info_list = []
        full_indice = np.arange(X.shape[0])

        for feature in range(X.shape[1]):

            # Get the threshold candidates
            threshold_candidates = []
            feature_values = np.sort(np.unique(X[:, feature]))
            for idx in range(feature_values.size - 1):
                threshold_value = (feature_values[idx] + feature_values[idx + 1]) / 2
                threshold_candidates = np.append(threshold_candidates, threshold_value)

            # Find the classification error for each threshold value
            for threshold in threshold_candidates:
                left_indice = full_indice[
                    np.argwhere(X[:, feature] <= threshold).flatten()
                ]
                right_indice = full_indice[
                    np.argwhere(X[:, feature] > threshold).flatten()
                ]

                # If left and right sizes are the same, take -1 as majority
                left_pred = (
                    [-1 for n in range(left_indice.size)]
                    if np.argwhere(y[left_indice] == -1).flatten().size
                    >= left_indice.size / 2
                    else [1 for n in range(left_indice.size)]
                )

                right_pred = (
                    [-1 for n in range(right_indice.size)]
                    if np.argwhere(y[right_indice] == -1).flatten().size
                    >= right_indice.size / 2
                    else [1 for n in range(right_indice.size)]
                )

                splitted = np.concatenate((y[left_indice], y[right_indice]))
                predicted = np.concatenate((left_pred, right_pred))

                classification_error = self.compute_error(splitted, predicted)

                decision_info = dict()
                decision_info["feature"] = feature
                decision_info["threshold"] = threshold
                decision_info["error"] = classification_error
                decision_info["left"] = left_pred[0]
                decision_info["right"] = right_pred[0]

                decision_info_list.append(decision_info)

        optimal_decision_info = min(decision_info_list, key=lambda x: x["error"])

        self.selected_feature = optimal_decision_info["feature"]
        self.threshold = optimal_decision_info["threshold"]
        self.left_prediction = optimal_decision_info["left"]
        self.right_prediction = optimal_decision_info["right"]

        #################

    def compute_error(self, pred, y):
        """
        Description:
            Compute the error using quality metric in .ipynb file.
        
        Args:
            pred: (N, ) numpy array. Prediction of decision stump.
            y: (N, ) numpy array. Target variable, has the values of 1 or -1.
                where N is the number of samples and D is the feature dimension.
            
        Returns:
            out: (float)
            
        """
        ### CODE HERE ###

        out = 1 - np.average(pred == y)

        #################
        return out

    def predict(self, X):
        """
        Description:
            Predict the target variables. Use the attributes.
        
        Args:
            X: (N, D) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of decision stump.
            
        """
        ### CODE HERE ###

        pred = np.zeros(X.shape[0])

        for idx in range(X.shape[0]):
            if X[idx, self.selected_feature] <= self.threshold:
                pred[idx] = self.left_prediction
            else:
                pred[idx] = self.right_prediction

        #################
        return pred


class AdaBoost:
    """AdaBoost class"""

    def __init__(self, num_estimators):
        """
        Description:
            Set the attributes. 
                
                num_estimator: int.
                error_history: list. List of weighted error history.
                classifiers: list. List of weak classifiers.
                             The items of classifiers (i.e., classifiers[1]) is the dictionary denoted as classifier.
                             The classifier has key 'coefficient' and 'classifier'. The values are the coefficient 
                             for that classifier and the Decsion stump classifier.

        
        Args:
            
        Returns:
            
        """
        np.random.seed(0)
        self.num_estimator = num_estimators
        self.classifiers = []
        self.error_history = []

    def fit(self, X, y):
        self.X = X
        self.y = y

        ### CODE HERE ###

        # initialize the data weight
        self.data_weight = np.ones(X.shape[0]) / X.shape[0]

        #################

        assert self.data_weight.shape == self.y.shape

        self.build_classifier()

    def build_classifier(self):
        """
        Description:
            Build adaboost classifier. Follow the procedures described in .ipynb file.
        
        Args:
            
        Returns:
            
        """
        ### CODE HERE ###

        full_indice = np.arange(self.X.shape[0])

        for stage in range(self.num_estimator):

            stump = DecisionStump()

            if stage == 0:
                sampled_idx = full_indice

                X_sampled = self.X
                y_sampled = self.y

            else:
                sampled_idx = np.random.choice(
                    full_indice, size=self.X.shape[0], p=self.data_weight, replace=True,
                )

                X_sampled = self.X[sampled_idx]
                y_sampled = self.y[sampled_idx]

            stump.fit(X_sampled, y_sampled)
            pred_unsorted = stump.predict(X_sampled)

            error = np.average(pred_unsorted != y_sampled)
            self.error_history.append(error)

            # pred = 0 if X[idx] is not sampled in this step
            pred = np.zeros(pred_unsorted.size)
            for i in range(pred_unsorted.size):
                pred[sampled_idx[i]] = pred_unsorted[i]

            coefficient = self.compute_classifier_coefficient(error)

            self.data_weight = self.update_weight(pred, coefficient)
            self.data_weight = self.normalize_weight()

            stage_classifier = dict()
            stage_classifier["classifier"] = stump
            stage_classifier["coefficient"] = coefficient

            self.classifiers.append(stage_classifier)

        #################

    def compute_classifier_coefficient(self, weighted_error):
        """
        Description:
            Compute the coefficient for classifier
        
        Args:
            weighted_error: numpy float. Weighted error for the classifier.
            
        Returns:
            coefficient: numpy float. Coefficient for classifier.
            
        """
        ### CODE HERE ###

        coefficient = np.log((1 - weighted_error) / weighted_error) / 2

        #################
        return coefficient

    def update_weight(self, pred, coefficient):
        """
        Description:
            Update the data weight. 
        
        Args:
            pred: (N, ) numpy array. Prediction of the weak classifier in one step.
            coefficient: numpy float. Coefficient for classifier.
            
        Returns:
            weight: (N, ) numpy array. Updated data weight.
            
        """
        ### CODE HERE ###

        weight = self.data_weight * np.exp(-coefficient * self.y * pred)

        #################
        return weight

    def normalize_weight(self):
        """
        Description:
            Normalize the data weight
        
        Args:
            
            
        Returns:
            weight: (N, ) numpy array. Normalized data weight.
            
        """
        ### CODE HERE ###

        weight = self.data_weight / np.sum(self.data_weight)

        #################
        return weight

    def predict(self, X):
        """
        Description:
            Predict the target variables (Adaboosts' final prediction). Use the attribute classifiers.
            
            Note that item of classifiers list should be a dictionary like below
                self.classfiers[0] : classifier,  (dict)
                
            The dictionary {key: value} is composed,
                classifier : {'coefficient': (coefficient value),
                              'classifier' : (decision stump classifier)}
        
        Args:
            X: (N, D) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of adaboost classifier. Output values are of 1 or -1.
            
        """
        ### CODE HERE ###

        sum_of_wf = np.zeros(X.shape[0])

        for n in range(self.num_estimator):
            coeff = self.classifiers[n]["coefficient"]
            stump = self.classifiers[n]["classifier"]

            weak_pred = stump.predict(X)
            sum_of_wf += weak_pred * coeff

        pred = sign(sum_of_wf)

        #################
        return pred

    def predict_proba(self, X):
        """
        Description:
            Predict the probabilities of prediction of each class using sigmoid function. The shape of the output is (N, number of classes)
        
        Args:
            X: (N, D) numpy array. Training/testing samples.
            
        Returns:
            proba: (N, number of classes) numpy array. Probabilities of adaboost classifier's decision.
            
        """
        ### CODE HERE ###

        proba = np.zeros((X.shape[0], 2))

        for n in range(self.num_estimator):
            coeff = self.classifiers[n]["coefficient"]
            stump = self.classifiers[n]["classifier"]

            weak_pred = stump.predict(X)
            proba[:, 0] += weak_pred * coeff

        proba[:, 0] = 1 / (1 + np.exp(proba[:, 0]))
        proba[:, 1] = 1 - proba[:, 0]

        #################
        return proba


def compute_staged_accuracies(classifier_list, X_train, y_train, X_test, y_test):
    """
        Description:
            Predict the accuracies over stages.
        
        Args:
            classifier_list: list of dictionary. Adaboost classifiers with coefficients.
            X_train: (N, D) numpy array. Training samples.
            y_train: (N, ) numpy array. Target variable, has the values of 1 or -1.
            X_test: (N', D) numpy array. Testing samples.
            y_test: (N', ) numpy array. Target variable, has the values of 1 or -1.
            
        Returns:
            acc_train: list. Accuracy on training samples. 
            acc_list: list. Accuracy on test samples.
                i.e, acc_train[40] =  $\hat{\mathbf{y}}=\text{sign} \left( \sum_{t=1}^{40} \hat{w_t} f_t(\mathbf{x}) \right)$
            
    """
    acc_train = []
    acc_test = []

    for i in range(len(classifier_list)):

        ### CODE HERE ###

        sum_train = np.zeros(X_train.shape[0])
        sum_test = np.zeros(X_test.shape[0])

        for n in range(i + 1):
            coeff = classifier_list[n]["coefficient"]
            stump = classifier_list[n]["classifier"]

            weak_pred_train = stump.predict(X_train)
            weak_pred_test = stump.predict(X_test)

            sum_train += weak_pred_train * coeff
            sum_test += weak_pred_test * coeff

        pred_train = sign(sum_train)
        pred_test = sign(sum_test)

        acc_train.append(np.average(pred_train == y_train))
        acc_test.append(np.average(pred_test == y_test))

        #################

    return acc_train, acc_test

