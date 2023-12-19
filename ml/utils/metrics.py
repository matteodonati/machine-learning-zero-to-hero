import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix.
    """
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           cm[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
    return cm

def accuracy_score(y_true, y_pred):
    """
    Computes the accuracy score.
    """
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.diag(cm)) / np.sum(cm)

def precision_score(y_true, y_pred):
    """
    Computes the precision score.
    """
    cm = confusion_matrix(y_true, y_pred)
    return np.mean(np.diag(cm) / np.sum(cm, axis=0))

def recall_score(y_true, y_pred):
    """
    Computes the recall score.
    """
    cm = confusion_matrix(y_true, y_pred)
    return np.mean(np.diag(cm) / np.sum(cm, axis=1))

def entropy(y):
    """
    Computes entropy.
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))

def gini_impurity(y):
    """
    Computes Gini impurity.
    """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p**2)

def information_gain(y_parent, y_left, y_right, criterion='gini'):
    """
    Computes information gain.
    """
    f = gini_impurity if criterion == 'gini' else entropy
    return f(y_parent) - (len(y_left) / len(y_parent)) * f(y_left) - (len(y_right) / len(y_parent)) * f(y_right)

def mean_squared_error(y_true, y_pred):
    """
    Mean squared error.
    """
    return np.mean(np.square(y_true - y_pred))