import numpy as np

class KMeans():
    """
    Implements the k-means clustering algorithm.
    """
    def __init__(self, n_clusters=2, n_iter=100):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.centers = None

    def fit(self, X):
        """
        Runs the k-means clustering algorithm on X.
        """
        n_samples = len(X)
        self.centers = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        for _ in range(self.n_iter):
            clusters = self.predict(X)
            new_centers = np.array([X[clusters == j].mean(axis=0) for j in range(self.n_clusters)])
            if np.all(self.centers == new_centers):
                break
            self.centers = new_centers

    def fit_predict(self, X):
        """
        Computes cluster centers and predicts the cluster index for each sample in X.
        """
        self.fit(X)
        return self.predict(X)

    def predict(self, X):
        """
        Predicts the closest cluster each sample in X belongs to.
        """
        distances = np.sqrt(((X[:, np.newaxis] - self.centers)**2).sum(axis=2))
        return np.argmin(distances, axis=1)

class DBSCAN():
    def __init__(self, eps=0.5, min_samples=5):
        """
        Implements the DBSCAN clustering algorithm.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        """
        Runs the DBSCAN clustering algorithm on X.
        """
        self.labels = [0] * len(X)
        label = 0
        for i in range(len(X)):
            if not (self.labels[i] == 0):
                continue
            neighbors = self._compute_neighborhood(X, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                label += 1
                self.labels[i] = label
                k = 0
                while k < len(neighbors):
                    neighbor_idx = neighbors[k]
                    if self.labels[neighbor_idx] == -1:
                        self.labels[neighbor_idx] = label
                    elif self.labels[neighbor_idx] == 0:
                        self.labels[neighbor_idx] = label
                        new_neighbors = self._compute_neighborhood(X, neighbor_idx)
                        if len(new_neighbors) >= self.min_samples:
                            neighbors += new_neighbors
                    k += 1

    def fit_predict(self, X):
        """
        Runs DBSCAN and returns the cluster index for each sample in X.
        """
        self.fit(X)
        return self.labels        

    def _compute_neighborhood(self, X, p):
        """
        Computes the neighborhood of a data point.
        """
        neighbors = []
        for i in range(len(X)):
            if np.linalg.norm(X[p] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors