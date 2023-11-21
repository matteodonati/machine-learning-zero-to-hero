import numpy as np
from ml.utils.metrics import information_gain

class Node():
    """
    Node of a decision tree model.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DecisionTreeClassifier():
    """
    Decision tree model.
    """
    def __init__(self, max_depth=2, min_samples_split=2):
        """
        Constructor.
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _best_split(self, data, n_samples, n_features):
        """
        Finds the best split according to the data.
        """
        feature = None
        threshold = None
        left = None
        right = None
        gain = -1.0

        for col in range(n_features):
            thresholds = np.unique(data[:, col])
            for curr_threshold in thresholds:
                curr_left = []
                curr_right = []
                for row in range(n_samples):
                    if data[row, col] <= curr_threshold:
                        curr_left.append(data[row])
                    else:
                        curr_right.append(data[row])
                curr_left = np.array(curr_left)
                curr_right = np.array(curr_right)
                if len(curr_left) != 0 and len(curr_right) != 0:
                    y = data[:, -1]
                    y_left = curr_left[:, -1]
                    y_right = curr_right[:, -1]
                    curr_gain = information_gain(y, y_left, y_right)
                    if curr_gain > gain:
                        feature = col
                        threshold = curr_threshold
                        left = curr_left
                        right = curr_right
                        gain = curr_gain
        return feature, threshold, left, right, gain

    def _build(self, data, depth=0):
        """
        Builds the decision tree.
        """
        X, y = data[:, :-1], data[:, -1]
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            feature, threshold, left, right, gain = self._best_split(data, n_samples, n_features)
            left = self._build(left, depth=depth + 1)
            right = self._build(right, depth=depth + 1)
            return Node(feature=feature, threshold=threshold, left=left, right=right, gain=gain)
        else:
            return Node(value=max(y, key=y.tolist().count))

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        self.root = self._build(np.column_stack((X, y)))

    def predict(self, X):
        """
        Predicts classes for X.
        """
        return [self.make_prediction(row, self.root) for row in X]
    
    def make_prediction(self, x, tree):
        """
        Predicts the class for a single data point.
        """
        if tree.value != None: 
            return tree.value
        value = x[tree.feature]
        if value <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)