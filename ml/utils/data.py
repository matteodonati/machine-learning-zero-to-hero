import numpy as np

def train_test_split(X, y, train_size=None, test_size=None):
    """
    Splits X and y into training and test portions.
    """
    n = len(X)
    p = np.random.permutation(n)
    X_shuffled, y_shuffled = X[p], y[p]
    if train_size != None:
        n_train = int(train_size * n)
    elif test_size != None:
        n_train = int(n - test_size * n)
    return X_shuffled[:n_train], X_shuffled[n_train:], y_shuffled[:n_train], y_shuffled[n_train:]

def normalize_data(X):
    """
    Normalizes the data.
    """
    for i in range(X.shape[-1]):
        col = X[:, i]
        X[:, i] = (col - col.min()) / (col.max() - col.min())
