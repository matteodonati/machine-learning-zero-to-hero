import numpy as np

def entropy(y):
    """
    Computes the entropy.
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))

def gini_impurity(y):
    """
    Computes the Gini impurity.
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p**2)

def information_gain(y_parent, y_left, y_right, criterion='gini'):
    """
    Computes the information gain.
    """
    f = gini_impurity if criterion == 'gini' else entropy
    return f(y_parent) - ((len(y_left) / len(y_parent)) * f(y_left) + (len(y_right) / len(y_parent)) * f(y_right))

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
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Constructor.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _best_split(self, X, y, n_samples, n_features):
        """
        Finds the best split according to the data.
        """
        pass

    def _get_leaf_value(self, y):
        """
        Computes the leaf value based on y.
        """
        return max(y, key=y.count)

    def _build(self, data, depth=0):
        """
        Builds the decision tree.
        """
        X, y = data[:, :-1], data[:, -1]
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            best = self._best_split(X, y, n_samples, n_features)
            if best["gain"] > 0:
                left = self._build(best["left"], depth + 1)
                right = self._build(best["right"], depth + 1)
                ...

    def fit(self, X, y):
        """
        Fits the decision tree to the data.
        """
        self._build(np.concatenate(X, y, axis=1))