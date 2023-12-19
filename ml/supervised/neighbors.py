import numpy as np

class KNeighborsClassifier():
    """
    K-Nearest Neighbors algorithm.
    """
    def __init__(self, n_neighbors=5):
        """
        Constructor.
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.labels = None

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        self.X_train = X
        self.y_train = y
        self.labels = np.unique(self.y_train)

    def predict(self, X):
        """
        Predicts classes for X.
        """
        y_pred = []
        dist = np.array([[np.linalg.norm(p - q) for q in self.X_train] for p in X])
        neighbors_idx = np.argpartition(dist, self.n_neighbors)
        for i in range(neighbors_idx.shape[0]):
            counts = dict((c, 0) for c in self.labels)
            neighbors_labels = self.y_train[neighbors_idx[i, :self.n_neighbors]]
            for j in range(neighbors_labels.shape[0]):
                counts[neighbors_labels[j]] += 1
            y_pred.append(max(counts, key=counts.get))
        return y_pred