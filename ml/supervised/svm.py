import numpy as np

class SVC:
    """
    Support Vector machine for classification.
    """
    def __init__(self, sigma=0.1, n_epochs=1000, lr=0.001):
        self.alpha = None
        self.b = 0
        self.C = 1
        self.sigma = sigma
        self.n_epochs = n_epochs
        self.lr = lr
        self.kernel = self._gaussian_kernel

    def _gaussian_kernel(self, X, Z):
        """
        Gaussian (RBF) kernel.
        """
        return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X[:, np.newaxis] - Z[np.newaxis, :], axis=2) ** 2)

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        y = np.where(y == 0, -1, 1)
        self.X = X
        self.y = y
        self.alpha = np.random.random(X.shape[0])
        self.b = 0
        self.ones = np.ones(X.shape[0])
        y_mul_kernel = np.outer(y, y) * self.kernel(X, X)
        for _ in range(self.n_epochs):
            gradient = self.ones - y_mul_kernel.dot(self.alpha)
            self.alpha += self.lr * gradient
            self.alpha = np.clip(self.alpha, 0, self.C)
        alpha_index = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        b_list = [y[index] - (self.alpha * y).dot(self.kernel(X, X[index])) for index in alpha_index]
        self.b = np.mean(b_list)

    def predict(self, X):
        """
        Predicts classes for X.
        """
        return np.where(np.sign((self.alpha * self.y).dot(self.kernel(self.X, X)) + self.b) == -1, 0, 1).tolist()