import numpy as np

class GaussianNB():
    """
    Gaussian Naive Bayes model.
    """
    def __init__(self, classes=None, priors=None, dist=None):
        self.classes = classes
        self.priors = priors
        self.dist = dist
        self.pdf = lambda x, mean, std : (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean)**2 / (2 * std**2)))

    def _compute_priors(self, y):
        """
        Computes the prior probability for all classes.
        """
        self.priors = dict((c, 0.0) for c in self.classes)
        for c in self.classes:
            self.priors[c] = len(np.where(y == c)[0]) / len(y)
    
    def _compute_dist(self, X, y):
        """
        Computes the parameters of Gaussian distributions that best fit the data.
        """
        self.dist = dict((c, {'mean': np.zeros(X.shape[-1]), 'std': np.zeros(X.shape[-1])}) for c in self.classes)
        for c in self.classes:
            self.dist[c]['mean'] = np.mean(X[y == c], axis=0)
            self.dist[c]['std'] = np.std(X[y == c], axis=0)
    
    def _compute_likelihoods(self, x):
        """
        Computes the likelihood probability for all classes.
        """
        likelihoods = dict((c, 1.0) for c in self.classes)
        for c in self.classes:
            for i in range(len(x)):
                likelihoods[c] *= self.pdf(x[i], self.dist[c]['mean'][i], self.dist[c]['std'][i])
        return likelihoods
    
    def _compute_posteriors(self, likelihoods):
        """
        Computes the posterior probability for all classes.
        """
        return dict((c, np.log(likelihoods[c] * self.priors[c])) for c in self.classes)

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        if self.classes == None:
            self.classes = np.unique(y)
        if self.priors == None:
            self._compute_priors(y)
        if self.dist == None:
            self._compute_dist(X, y)

    def predict(self, X):
        """
        Predicts classes for X.
        """
        y_pred = []
        for row in X:
            likelihoods = self._compute_likelihoods(row)
            posteriors = self._compute_posteriors(likelihoods)
            y_pred.append(max(posteriors, key=posteriors.get))
        return y_pred