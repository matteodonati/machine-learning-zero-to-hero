import numpy as np

class LogisticRegression():
    """
    Logistic Regression model.
    """
    def __init__(self, n_epochs=100, lr=1e-3):
        self.weights = None
        self.bias = 0.0
        self.n_epochs = n_epochs
        self.lr = lr

    def _sigmoid(self, Z):
        """
        Implements the stable version of the sigmoid function.
        """
        return np.array([1 / (1 + np.exp(-z)) if z >= 0 else np.exp(z) / (1 + np.exp(z)) for z in Z])
    
    def _loss(self, y_true, y_pred, eps=1e-9):
        """
        Implements the stable version of the binary cross-entropy loss function. Debug only.
        """
        return -1.0 * np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    
    def _update(self, X, y_true, y_pred):
        """
        Computes the gradient of the loss function and updates the model.
        """
        dLdw = np.matmul((y_pred - y_true), X)
        dLdb = np.sum((y_pred - y_true))
        self.weights = self.weights - self.lr * dLdw
        self.bias = self.bias - self.lr * dLdb

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        self.weights = np.zeros((X.shape[1]))
        for _ in range(self.n_epochs):
            z = np.matmul(self.weights, X.transpose()) + self.bias
            y_pred = self._sigmoid(z)
            self._update(X, y, y_pred)

    def predict(self, X):
        """
        Predicts classes for X.
        """
        z = np.matmul(self.weights, X.transpose()) + self.bias
        y_pred = self._sigmoid(z)
        return [1 if p > 0.5 else 0 for p in y_pred]