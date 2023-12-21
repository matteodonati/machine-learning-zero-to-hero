import streamlit as st

st.set_page_config(layout='centered')
st.title('Documentation :page_facing_up:')

CLASSIFICATION_TREE = 'Decision tree (classification)'
NAIVE_BAYES = 'Gaussian naive Bayes'
KNN = 'K-nearest neighbors (classification)'
LR = 'Logistic regression'
SVM = 'Support vector machine'
LINEAR_REGRESSION = 'Linear regression'
REGRESSION_TREE = 'Decision tree (regression)'
KNN_REGRESSION = 'K-nearest neighbors (regression)'
KMEANS = 'K-means'
DBSCAN = 'DBSCAN'

st.markdown(
    """
    This page offers documentation for all the implemented code. While it provides a concise 
    introduction to the various models and algorithms, it is recommended to supplement it 
    with more comprehensive resources for a deeper understanding. Use the following `selectbox`
    to access the documentation.

    <script
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
        type="text/javascript">
    </script>
    """,
    unsafe_allow_html=True
)

option = st.selectbox(
    'Select a documentation page',
    (CLASSIFICATION_TREE, NAIVE_BAYES, KNN, LR, SVM, LINEAR_REGRESSION, REGRESSION_TREE, KNN_REGRESSION, KMEANS, DBSCAN),
    index=None,
)

if option == CLASSIFICATION_TREE:
    st.markdown(
        """
        ---
        
        ## Classification Tree <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/tree.py" style="font-size: 15px">[source]</a>

        A decision tree for classification problems is a predictive model that maps features 
        (input variables) to outcomes (classes or labels) by recursively splitting 
        the data based on the values of the features. It is a tree-like structure where 
        each internal node represents a decision based on a particular feature, each 
        branch represents an outcome of that decision, and each leaf node represents 
        the final class label. The goal of a decision tree is to create a set of rules 
        that can be easily followed to make predictions about the class of a new instance. 
        The process of building a decision tree involves selecting the best feature at 
        each node to maximize the separation of classes in the data.
        
        The following code defines the classes required to implement a decision 
        tree model for classification. This documentation will provide an overview 
        of how the proposed implementation works and describe the key components of 
        the code.

        ### `Node` Class

        The `Node` class represents a node in the decision tree. Each node is 
        characterized by the following attributes:

        - `feature`: The feature used for splitting data at this node.
        - `threshold`: The threshold value for the feature.
        - `left`: The left child node.
        - `right`: The right child node.
        - `gain`: Information gain achieved by splitting at this node.
        - `value`: The class label (for leaf nodes) or None (for non-leaf nodes).

        ```python
        class Node():
            def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
                self.feature = feature
                self.threshold = threshold
                self.left = left
                self.right = right
                self.gain = gain
                self.value = value
        ```

        ### `DecisionTreeClassifier` Class

        The `DecisionTreeClassifier` class represents the decision tree model itself. 
        It includes the following methods and attributes:

        #### Constructor

        The constructor initializes the decision tree with hyperparameters such as 
        `max_depth` (maximum depth of the tree) and `min_samples_split` (minimum number 
        of samples required to split a node).

        ```python
        class DecisionTreeClassifier:
            def __init__(self, max_depth=2, min_samples_split=2):
                self.root = None
                self.max_depth = max_depth
                self.min_samples_split = min_samples_split
        ```

        #### `_best_split` Method

        The `_best_split` method is responsible for finding the best feature and threshold 
        combination for splitting the data based on information gain. Let's break down 
        the code step by step:

        - The method starts by initializing variables to keep track of the best feature, 
        threshold, left and right data subsets, and the information gain.

        ```python
        def _best_split(self, data, n_samples, n_features):
            feature = None
            threshold = None
            left = None
            right = None
            gain = -1.0
        ```

        - It then iterates through the features (columns) of the dataset, represented by 
        the variable `col`. For each feature, it calculates the unique thresholds present 
        in that feature using `np.unique(data[:, col])`.

        ```python
        for col in range(n_features):
            thresholds = np.unique(data[:, col])
            for curr_threshold in thresholds:
        ```

        - Within the threshold loop, it creates two lists, `curr_left` and `curr_right`, to 
        collect data points that satisfy and do not satisfy the current threshold condition.

        ```python
        curr_left = []
        curr_right = []
        ```

        - It then iterates through the rows of the data, represented by the variable `row`, 
        and assigns each data point to either `curr_left` or `curr_right` based on whether it 
        satisfies the threshold condition.

        ```python
        for row in range(n_samples):
            if data[row, col] <= curr_threshold:
                curr_left.append(data[row])
            else:
                curr_right.append(data[row])
        ```

        - The method checks if both `curr_left` and `curr_right` have at least one data point 
        (i.e., their lengths are not zero) to ensure that there is a valid split. It then calculates 
        information gain (`curr_gain`) by calling the `information_gain` function with appropriate 
        arguments.

        ```python
        if len(curr_left) != 0 and len(curr_right) != 0:
            y = data[:, -1]
            y_left = curr_left[:, -1]
            y_right = curr_right[:, -1]
            curr_gain = information_gain(y, y_left, y_right)
        ```

        - If the current gain is greater than the previously recorded best gain, the method 
        updates the variables for the best feature, threshold, and data subsets.

        ```python
        if curr_gain > gain:
            feature = col
            threshold = curr_threshold
            left = curr_left
            right = curr_right
            gain = curr_gain
        ```

        - Finally, the method returns the best feature, threshold, left and right data subsets, 
        and the associated gain.

        ```python
        return feature, threshold, left, right, gain
        ```

        #### `_build` Method

        The `_build` method is responsible for constructing the decision tree. Here's a breakdown 
        of the code:

        - The method starts by extracting the feature matrix `X` and label vector `y` from the input 
        data. It then calculates the number of samples (`n_samples`) and the number of features (`n_features`) 
        in the feature matrix.

        ```python
        def _build(self, data, depth=0):
            X, y = data[:, :-1], data[:, -1]
            n_samples, n_features = X.shape
        ```

        - The method checks two conditions to determine whether further splitting is allowed:
            - `n_samples` must be greater than or equal to the `min_samples_split` hyperparameter.
            - The current depth (`depth`) must be less than or equal to the `max_depth` hyperparameter.
        - If both conditions are met, the method proceeds with further splitting:
            - It calls the `_best_split` method to find the best feature and threshold for the 
            current data.
            - It then recursively calls `_build` on the left and right data subsets (`left` and `right`) 
            and increments the depth by one.
            - A new `Node` is created with the best feature, threshold, left and right child nodes, 
            and the associated gain.

        ```python
        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            feature, threshold, left, right, gain = self._best_split(data, n_samples, n_features)
            left = self._build(left, depth=depth + 1)
            right = self._build(right, depth=depth + 1)
            return Node(feature=feature, threshold=threshold, left=left, right=right, gain=gain)
        ```

        - If the stopping conditions are not met, indicating that further splitting is not allowed, 
        the method creates a leaf node. It calculates the majority class in the current node's data 
        using `max(y, key=y.tolist().count)` and assigns it as the value of the leaf node.

        ```python
        else:
            return Node(value=max(y, key=y.tolist().count))
        ```

        #### `fit` Method

        The `fit` method is used for training the decision tree on a dataset. Here's how it works:

        - The `fit` method takes two parameters: `X` (features) and `y` (labels).
        - It assigns the result of the `_build` method to `self.root`. This is where the decision 
        tree construction process starts.
        - It calls `_build` with the entire dataset, created by stacking the feature matrix `X` and 
        label vector `y` horizontally using `np.column_stack((X, y))`.

        ```python
        def fit(self, X, y):
            self.root = self._build(np.column_stack((X, y)))
        ```

        #### `predict` Method

        The `predict` method is used to make predictions for a dataset. Here's how it's implemented:

        - The `predict` method takes one parameter: `X`, which is the dataset for which predictions 
        are to be made.
        - It uses a list comprehension to make predictions for each data point in `X`. For each data 
        point (row) in `X`, it calls the `make_prediction` method and passes the data point and the root 
        node of the decision tree as arguments.
        - The method returns a list of predicted class labels for each data point in `X`.

        ```python
        def predict(self, X):
            return [self.make_prediction(row, self.root) for row in X]
        ```

        #### `make_prediction` Method

        The `make_prediction` method is a recursive function used to predict the class of a single 
        data point. Here's how it works:

        - The `make_prediction` method takes two parameters: `x` (the data point to predict) and `tree` 
        (the current node being considered in the decision tree).
        - It first checks if the current node is a leaf node by examining the value attribute. If 
        the value is not `None`, it means it's a leaf node, and that value is returned as the 
        prediction.
        - If the current node is not a leaf node, it compares the feature value of the data point 
        `x` to the threshold in the current node. Depending on the comparison, it recursively calls 
        itself on either the left or right child node until it reaches a leaf node.

        ```python
        def make_prediction(self, x, tree):
            if tree.value != None: 
                return tree.value
            value = x[tree.feature]
            if value <= tree.threshold:
                return self.make_prediction(x, tree.left)
            else:
                return self.make_prediction(x, tree.right)
        ```
        """,
        unsafe_allow_html=True
    )
elif option == NAIVE_BAYES:
    st.markdown(
        """
        ---
        
        ## Gaussian Naive Bayes <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/naive_bayes.py" style="font-size: 15px">[source]</a>
        
        The Gaussian Naive Bayes algorithm is a probabilistic classification 
        model based on Bayes' theorem with the assumption of independence between 
        features. Bayes' theorem states that
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        P(c_k \vert \mathbf{x}) = \frac{P(c_k) P(\mathbf{x} \vert c_k)}{P(\mathbf{x})}
    """)
    st.markdown(
        """
        where $$c_k$$ is the $$k$$-th class in a vector $$C$$ of possible classes, 
        and $$\mathbf{x}$$ is an input sample (i.e., a vector of features). Here,
        $$P(c_k)$$ is the *prior* for class $$c_k$$, $$P(\mathbf{x} | c_k)$$ is 
        called *likelihood* and defines the probability of observing $$\mathbf{x}$$ 
        given class $$c_k$$, $$P(\mathbf{x})$$ is the *evidence* for $$\mathbf{x}$$, 
        and $$P(c_k | \mathbf{x})$$ is the *posterior* probability of observing class
        $$c_k$$ given the input $$\mathbf{x}$$. In this setting, there is no interest
        in the evidence, since it does not depend on $$C$$: 
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        P(c_k \vert \mathbf{x}) \propto P(c_k) P(\mathbf{x} \vert c_k)
    """)
    st.markdown(
        """
        Moreover, using the definition of conditional probability and applying the
        chain rule, $$P(c_k) P(\mathbf{x} | c_k)$$ can be expressed as
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        \begin{align*}
            P(c_k) P(\mathbf{x} \vert c_k) &= P(\mathbf{x}, c_k) \\
             &= P(x_1, x_2, \dots, c_k) \\
             &= P(x_1 \vert x_2, \dots, x_n, c_k) P(x_2, \dots, x_n, c_k) \\
             &= \dots \\
             &= P(x_1 \vert x_2, \dots, x_n, c_k) P(x_2 \vert x_3, \dots, x_n, c_k) \cdots P(x_n \vert c_k) P(c_k)
        \end{align*}
    """)
    st.markdown(
        """
        Naive Bayes makes the important assumption of independent features. Thus,
        $$P(x_i | x_{i + 1}, \dots x_n, c_k) $$ $$= P(x_i | c_k) $$, allowing
        the posterior probability to be computed as
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        \begin{align*}
            P(c_k \vert \mathbf{x}) &\propto P(\mathbf{x}, c_k) \\
             & \;\;\;\; = P(c_k) P(x_1 \vert c_k) \cdots P(x_n \vert c_k) \\
             & \;\;\;\; = P(c_k) \prod_{i=1}^{n} P(x_i \vert c_k)
        \end{align*}
    """)
    st.markdown(
        """
        In other words, given an input sample $$\mathbf{x}$$, the predicted class
        $$\hat{y}$$ will be computed as
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        \hat{y} = \operatorname{argmax}_{k \in 1, \dots, K} P(c_k) \prod_{i=1}^{n} P(x_i \vert c_k)
    """)
    st.markdown(
        """
        In the case of Gaussian naive Bayes, the probability density of value $$v$$
        given a class $$c_k$$ is computed as
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        P(x = v \vert c_k) = \frac{1}{\sqrt{2 \pi \sigma^2_k}} \exp\left(\frac{(v - \mu_k)^2}{2 \sigma^2_k}\right)
    """)
    st.markdown(
        """
        where $$\mu_k$$ and $$\sigma_k$$ have to be computed for each feature,
        considering training data.

        The following code defines the class required to implement Gaussian naive 
        Bayes. This documentation will provide an overview of how the proposed 
        implementation works and describe the key components of the code.

        ### `GaussianNB` Class

        The `GaussianNB` class implements the Gaussian naive Bayes algorithm.
        It includes the following methods and attributes:

        #### Constructor

        The constructor initializes the class defining different attributes: 
        - `classes`, the list of possible classes. 
        - `priors`, prior probability for each class.
        - `dist`, $$\mu$$ and $$\sigma$$ of the different distributions.
        - `pdf`, the probability density function of the Gaussian distribution.

        ```python
        class GaussianNB:
            def __init__(self):
                self.classes = None
                self.priors = None
                self.dist = None
                self.pdf = lambda x, mean, std : (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean)**2 / (2 * std**2)))
        ```
        
        #### `_compute_priors` Method

        This method computes the prior probabilities for each class based on the 
        frequency of class occurrences in the training labels.

        ```python
        def _compute_priors(self, y):
            self.priors = dict((c, 0.0) for c in self.classes)
            for c in self.classes:
                self.priors[c] = len(np.where(y == c)[0]) / len(y)
        ```

        #### `_compute_dist` Method

        This method computes the mean and standard deviation for each feature in 
        each class, forming the parameters of Gaussian distributions.

        ```python
        def _compute_dist(self, X, y):
            self.dist = dict((c, {'mean': np.zeros(X.shape[-1]), 'std': np.zeros(X.shape[-1])}) for c in self.classes)
            for c in self.classes:
                self.dist[c]['mean'] = np.mean(X[y == c], axis=0)
                self.dist[c]['std'] = np.std(X[y == c], axis=0)
        ```

        #### `_compute_likelihoods` Method

        This method computes the likelihood probabilities for each class given 
        a specific data point.

        ```python
        def _compute_likelihoods(self, x):
            likelihoods = dict((c, 1.0) for c in self.classes)
            for c in self.classes:
                for i in range(len(x)):
                    likelihoods[c] *= self.pdf(x[i], self.dist[c]['mean'][i], self.dist[c]['std'][i])
            return likelihoods
        ```

        #### `_compute_posteriors` Method

        This method computes the posterior probabilities for each class based 
        on the likelihoods and priors. The logarithm function is used to
        prevent underflows.

        ```python
        def _compute_posteriors(self, likelihoods):
            return dict((c, np.log(likelihoods[c] * self.priors[c])) for c in self.classes)
        ```

        #### `fit` Method

        This method fits the Gaussian Naive Bayes model to the training data by 
        setting possible class labels, computing priors, and distribution parameters 
        if not provided.

        ```python
        def fit(self, X, y):
            if self.classes == None:
                self.classes = np.unique(y)
            if self.priors == None:
                self._compute_priors(y)
            if self.dist == None:
                self._compute_dist(X, y)
        ```

        #### `predict` Method

        This method predicts class labels for a given set of data points using 
        likelihoods and posteriors.

        ```python
        def predict(self, X):
            y_pred = []
            for row in X:
                likelihoods = self._compute_likelihoods(row)
                posteriors = self._compute_posteriors(likelihoods)
                y_pred.append(max(posteriors, key=posteriors.get))
            return y_pred
        ```
        """,
        unsafe_allow_html=True
    )
elif option == KNN:
    st.markdown(
        """
        ---

        ## $$k$$-Nearest Neighbors (Classification) <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/neighbors.py" style="font-size: 15px">[source]</a>

        The $$k$$-nearest neighbors ($$k$$NN) algorithm is a simple and effective supervised learning 
        algorithm used for classification and regression. It classifies a data point based on 
        the majority class among its $$k$$-nearest neighbors. 
        
        The following code defines the class required to implement $$k$$NN. This documentation will 
        provide an overview of how the proposed implementation works and describe the key components 
        of the code.

        ### `KNeighborsClassifier` Class

        The `KNeighborsClassifier` class implements the kNN algorithm for classification. Let's go 
        through the key components of the code.

        #### Constructor

        Initializes the $$k$$NN classifier with the number of neighbors to consider (`n_neighbors`).

        ```python
        class KNeighborsClassifier:
            def __init__(self, n_neighbors=5):
                self.n_neighbors = n_neighbors
                self.X_train = None
                self.y_train = None
                self.labels = None
        ```

        #### `fit` Method

        Fits the $$k$$NN model to the provided training data. This algorithm is considered to
        be a *lazy* algorithm, meaning that all computation occurs at test time.

        ```python
        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            self.labels = np.unique(self.y_train)
        ```

        #### `predict` Method

        Predicts class labels for the given data points. 
        
        - This method first computes the Euclidean distance between each point in X and all 
        points in the training data:
        
        ```python
        def predict(self, X):
            y_pred = []
            dist = np.array([[np.linalg.norm(p - q) for q in self.X_train] for p in X])
        ```

        - It then identifies the indices of the $$k$$-nearest neighbors for each point, 
        and counts the occurrences of each class label among the neighbors:

        ```python
        neighbors_idx = np.argpartition(dist, self.n_neighbors)
        for i in range(neighbors_idx.shape[0]):
            counts = dict((c, 0) for c in self.labels)
            neighbors_labels = self.y_train[neighbors_idx[i, :self.n_neighbors]]
            for j in range(neighbors_labels.shape[0]):
                counts[neighbors_labels[j]] += 1
        ```
        
        - Lastly, it predicts the class label with the maximum count for each point:

        ```python
        y_pred.append(max(counts, key=counts.get))
        ```
        """,
        unsafe_allow_html=True
    )
elif option == LR:
    st.markdown(
        """
        ---

        ## Logistic Regression <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/linear.py" style="font-size: 15px">[source]</a>

        Logistic regression is a method used for binary classification problems, where 
        the goal is to predict the probability of an instance belonging to a particular 
        class. Despite its name, logistic regression is used for classification rather 
        than regression tasks.

        The logistic function, also known as the sigmoid function, transforms any real-valued 
        number into a value between $$0$$ and $$1$$. In logistic regression, this function is applied 
        to a linear combination of the input features creating a model that outputs probabilities: 
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        \sigma(z) = \sigma(\mathbf{w} \cdot \mathbf{x} + b) = 
             \begin{cases}
             \dfrac{1}{(1 + \exp(-(\mathbf{w} \cdot \mathbf{x} + b)))} & \text{if $(\mathbf{w} \cdot \mathbf{x} + b) \geq 0$} \\[4mm]
             \dfrac{\exp(\mathbf{w} \cdot \mathbf{x} + b)}{(1 + \exp(\mathbf{w} \cdot \mathbf{x} + b))} & \text{otherwise}
             \end{cases}
    """)
    st.markdown(
        """
        The predicted probability can then be converted into a binary outcome by applying 
        a threshold (commonly $$0.5$$). Instances with a probability greater than or equal 
        to the threshold are assigned to one class, while those below the threshold are assigned 
        to the other class.

        The following code defines the class required to implement logistic regression. This 
        documentation will provide an overview of how the proposed implementation works and 
        describe the key components of the code.

        ### `LogisticRegression` Class

        The `LogisticRegression` class implements a simple logistic regression model.

        #### Constructor

        Initializes the logistic regression model with the following parameters:

        - `weights`, the weights of the model.
        - `bias`, the bias of the model.
        - `n_epochs`, the number of training epochs.
        - `lr`, the learning rate.

        ```python
        class LogisticRegression():
            def __init__(self, n_epochs=100, lr=1e-3):
                self.weights = None
                self.bias = 0.0
                self.n_epochs = n_epochs
                self.lr = lr
        ```
        
        #### `_sigmoid` Method

        Implements a stable version of the sigmoid activation function to prevent numerical 
        instability.

        ```python
        def _sigmoid(self, Z):
            return np.array([1 / (1 + np.exp(-z)) if z >= 0 else np.exp(z) / (1 + np.exp(z)) for z in Z])
        ```

        #### `_loss` Method

        Implements a stable version of the binary cross-entropy loss function. Binary cross-entropy
        is defined as follows:
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        \mathcal{L}_{BCE}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{N} \sum_{i=0}^{N} \mathbf{y} \log(\hat{\mathbf{y}}) + (1 - \mathbf{y}) \log(1 - \hat{\mathbf{y}})
    """)
    st.markdown(
        """
        where $$N$$ is the number of samples.

        ```python
        def _loss(self, y_true, y_pred, eps=1e-9):
            return -1.0 * np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        ```

        #### `_update` Method

        Computes the gradient of the loss function and updates the model parameters using 
        gradient descent.

        ```python
        def _update(self, X, y_true, y_pred):
            dLdw = np.matmul((y_pred - y_true), X)
            dLdb = np.sum((y_pred - y_true))
            self.weights = self.weights - self.lr * dLdw
            self.bias = self.bias - self.lr * dLdb
        ```

        In particular, `dLdw` and `dLdb` are the partial derivatives of the loss function with 
        respect to the weights and the bias, respectively.

        #### `fit` Method

        Fits the logistic regression model to the provided training data:

        - It first computes the activation z of the model.
        - It then predicts output probabilities using the sigmoid function.
        - It lastly update the model's parameters using gradient descent.

        ```python
        def fit(self, X, y):
            self.weights = np.zeros((X.shape[1]))
            for _ in range(self.n_epochs):
                z = np.matmul(self.weights, X.transpose()) + self.bias
                y_pred = self._sigmoid(z)
                self._update(X, y, y_pred)
        ```

        #### `predict` Method

        Predicts class labels for the given data points.

        ```python
        def predict(self, X):
            z = np.matmul(self.weights, X.transpose()) + self.bias
            y_pred = self._sigmoid(z)
            return [1 if p > 0.5 else 0 for p in y_pred]
        ```
        """,
        unsafe_allow_html=True
    )
elif option == SVM:
    st.markdown(
        """
        ---

        ## Support Vector Machine <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/svm.py" style="font-size: 15px">[source]</a>

        Support Vector Machines (SVMs) are a set of supervised learning methods used for 
        classification, regression, and outlier detection. An SVM with an RBF kernel aims 
        to find a decision boundary in a transformed feature space where the classes can 
        be separated by a hyperplane. The decision function is given by:

        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        f(x) = \operatorname{sign}\left(\sum_{i = 1}^n \alpha_i y_i K(x_i, x) + b \right)
    """)
    st.markdown(
        """
        where:

        - $$K$$ is the RBF kernel function.
        - $$\\alpha_i$$ are the Langrange multipliers.
        - $$y_i$$ are the class labels.
        - $$b$$ is the bias term.
        - $$n$$ is the number of support vector.

        The objective is to maximize the margin between the two classes while penalizing 
        misclassifications, controlled by the hyperparameter C. The optimization problem 
        is solved using an iterative approach, typically SMO, to find the optimal values of
        $$\\alpha$$ and $$b$$.

        This documentation covers the implementation of an SVM classifier using the RBF 
        kernel.

        ### `SVC` Class

        The `SVC` (Support Vector Classifier) class implements an SVM classifier with an 
        RBF kernel. Below is a detailed explanation of its components.

        #### Constructor

        The constructor initializes the SVM classifier with hyperparameters and the 
        Gaussian kernel function.

        ```python
        class SVC:
            def __init__(self, sigma=0.1, n_epochs=1000, lr=0.001):
                self.alpha = None
                self.b = 0
                self.C = 1
                self.sigma = sigma
                self.n_epochs = n_epochs
                self.lr = lr
                self.kernel = self._gaussian_kernel
        ```

        In particular:

        - `sigma` controls the width of the Gaussian kernel.
        - `n_epochs` is the number of iterations for training.
        - `lr` is the learning rate.
        - `C` ($$C$$) is the regularization parameter. The strength of regularization is inversely 
        proportional to such parameter.
        - `alpha` are the Lagrange multipliers, used for optimization.
        - `b` is the intercept term in the decision function.
        - `kernel` is the kernel function used in the algorithm.

        #### `_gaussian_kernel` Method

        This method implements the Gaussian (RBF) kernel.

        ```python
        def _gaussian_kernel(self, X, Z):
            return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X[:, np.newaxis] - Z[np.newaxis, :], axis=2) ** 2)
        ```

        The kernel computes the exponential of the negative squared Euclidean distance 
        between vectors, normalized by `sigma`.

        #### `fit` Method
        
        The `fit` method trains the SVM model on the provided dataset. It takes the feature 
        matrix `X` and the label vector `y` as inputs and adjusts the model's parameters based 
        on these inputs.

        First,

        ```python
        def fit(self, X, y):
            y = np.where(y == 0, -1, 1)
        ```

        converts the label vector `y` so that its elements are either $$-1$$ or $$1$$. This is a 
        standard practice in SVM, as it simplifies the mathematical formulation of the problem.

        Then,

        ```python
        self.alpha = np.random.random(X.shape[0])
        self.b = 0
        self.ones = np.ones(X.shape[0])
        ```

        initializes the Lagrange multipliers $$\\alpha$$ with random values and sets the bias term 
        $$b$$ to zero. It also creates a helper vector, consisting of ones, which is used in the 
        gradient calculation.

        ```python
        y_mul_kernel = np.outer(y, y) * self.kernel(X, X)
        ```

        Computes the kernel matrix once and stores it in `y_mul_kernel`. This matrix represents 
        the pairwise kernel calculations between all samples in `X`. The matrix is scaled by 
        the outer product of the label vector with itself, which is a common step in SVM calculations.

        ```python
        for _ in range(self.n_epochs):
            gradient = self.ones - y_mul_kernel.dot(self.alpha)
            self.alpha += self.lr * gradient
            self.alpha = np.clip(self.alpha, 0, self.C)
        ```

        Iteratively adjusts the values of $$\\alpha$$ for a specified number of epochs. The gradient of 
        the objective function with respect to $$\\alpha$$ is computed. This gradient is used to update 
        $$\\alpha$$. The values of $$\\alpha$$ are then clipped to ensure they remain within the $$[0, C]$$ 
        interval, as per the box constraints of the SVM optimization problem.

        Lastly,

        ```python
        alpha_index = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        b_list = [y[index] - (self.alpha * y).dot(self.kernel(X, X[index])) for index in alpha_index]
        self.b = np.mean(b_list)
        ```

        identifies support vectors (those for which $$\\alpha$$ is neither $$0$$ nor $$C$$) and uses them to compute 
        the bias term $$b$$. For each support vector, calculates an estimate of $$b$$. This is done by 
        considering the conditions that support vectors should satisfy in the optimal solution.
        The final bias term is set as the mean of these estimates, providing a robust estimate 
        even in the presence of noise.

        #### `predict` Method

        The `predict` method makes predictions for new data.

        ```python
        def predict(self, X):
            return np.where(np.sign((self.alpha * self.y).dot(self.kernel(self.X, X)) + self.b) == -1, 0, 1).tolist()
        ```

        Predictions are made based on the sign of the decision function, which involves the learned 
        $$\\alpha$$, $$b$$, and the kernel function.
        """,
        unsafe_allow_html=True
    )
elif option == LINEAR_REGRESSION:
    st.markdown(
        """
        ---

        ## Linear Regression <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/linear.py" style="font-size: 15px">[source]</a>

        The following class is an implementation of the linear regression model, 
        a fundamental algorithm in machine learning and statistics for predicting 
        a continuous target variable based on one or more explanatory variables.
        The linear regression model assumes a linear relationship between the 
        independent variable X and the dependent variable y. This relationship 
        is represented as:
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        y = mX + c
    """)
    st.markdown(
        """
        where $$y$$ is the dependent variable, $$X$$ is the independent variable,
        $$m$$ is the slope of the line, and $$c$$ is the $$y$$-axis intercept.

        The goal of linear regression is to find the best values for $$m$$ and $$c$$ such 
        that the predicted values of $$y$$ are as close as possible to the actual values. 
        This is typically done by minimizing the mean squared error (MSE), a common loss 
        function in regression problems:
        """,
        unsafe_allow_html=True
    )
    st.latex(r"""
        MSE = \frac{1}{N} \sum_{i = 0}^{N}(y_i - \hat{y}_i)^2
    """)
    st.markdown(
        """
        where $$N$$ is the number of observations, $$y_i$$ is the $$i$$-th ground truth
        value of the target variable, and $$\hat{y}_i$$ is the predicted value for the
        target variable.

        In the context of this implementation, gradient descent is used to minimize the
        loss function.

        This implementation is designed to handle simple linear regression with a single feature.

        ### `LinearRegression` Class

        The `LinearRegression` class is used to create a linear model that predicts the dependent 
        variable ($$y$$) based on the values of the independent variable ($$X$$). It assumes a 
        linear relationship between the two variables.

        #### Constructor

        The constructor method initializes the Linear Regression model with default settings.

        ```python
        class LinearRegression():
            def __init__(self, n_epochs=2000, lr=1e-3):
                self.m = 0.0
                self.c = 0.0
                self.n_epochs = n_epochs
                self.lr = lr
        ```

        `n_epochs` is the number of iterations for the training process, while `lr` is the learning 
        rate, determining the step size at each iteration while moving toward a minimum of the 
        loss function. The model parameters (`m` and `c`) are initialized to zero.

        #### `_update` Method

        A private method used during the fitting process to update the model's parameters based 
        on the gradient of the loss function.

        ```python
        def _update(self, X, y, y_pred):
            dLdm = (-2 / len(X)) * np.sum(X * (y - y_pred))
            dLdc = (-2 / len(X)) * np.sum(y - y_pred)
            self.m = self.m - self.lr * dLdm
            self.c = self.c - self.lr * dLdc
        ```

        The method performs the following steps:
        1. It computes the gradients of the loss function with respect to `m` (`dLdm`) and c (`dLdc`).
        2. It updates the values of `m` and `c` using gradient descent, where the parameters are 
        adjusted in the opposite direction of the gradient.

        #### `fit` Method

        The `fit` method is used to train the linear regression model on the provided dataset.

        ```python
        def fit(self, X, y):
            for _ in range(self.n_epochs):
                y_pred = self.m * X + self.c
                self._update(X, y, y_pred)
        ```

        The method iteratively adjusts `m` and `c` to minimize the loss function. It does this 
        over the number of epochs (`n_epochs`) specified during initialization.

        #### `predict` Method

        After the model is trained, the `predict` method can be used to make predictions on new 
        data.

        ```python
        def predict(self, X):
            return self.m * X + self.c
        ```

        This method calculates the predicted target values as a linear combination of the input 
        `X` and the learned parameters `m` and `c`.
        """,
        unsafe_allow_html=True
    )
elif option == REGRESSION_TREE:
    st.markdown(
        """
        ---
        
        ## Regression Tree <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/tree.py" style="font-size: 15px">[source]</a>

        A decision tree for regression problems is a predictive model that maps features 
        (input variables) to continuous outcomes (target values) by recursively splitting 
        the data based on the values of the features. It is a tree-like structure where 
        each internal node represents a decision based on a particular feature, each branch 
        represents an outcome of that decision, and each leaf node represents a predicted 
        value for the dependent variable. The goal of a regression tree is to create a set 
        of rules that can be easily followed to predict the value of a new instance. The 
        process of building a regression tree involves selecting the best feature at each 
        node to minimize the variance of the target values within each partition of the data.
        
        The following code defines the classes required to implement a decision 
        tree model for regression. This documentation will provide an overview 
        of how the proposed implementation works and describe the key components of 
        the code.

        ### `Node` Class

        The `Node` class represents a node in the decision tree. Each node is 
        characterized by the following attributes:

        - `feature`: The feature used for splitting data at this node.
        - `threshold`: The threshold value for the feature.
        - `left`: The left child node.
        - `right`: The right child node.
        - `gain`: Information gain achieved by splitting at this node.
        - `value`: The output value (for leaf nodes) or None (for non-leaf nodes).

        ```python
        class Node():
            def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
                self.feature = feature
                self.threshold = threshold
                self.left = left
                self.right = right
                self.gain = gain
                self.value = value
        ```

        ### `DecisionTreeRegressor` Class

        The `DecisionTreeRegressor` class represents the decision tree model itself. 
        It includes the following methods and attributes:

        #### Constructor

        The constructor initializes the decision tree with hyperparameters such as 
        `max_depth` (maximum depth of the tree) and `min_samples_split` (minimum number 
        of samples required to split a node).

        ```python
        class DecisionTreeRegressor:
            def __init__(self, max_depth=2, min_samples_split=2):
                self.root = None
                self.max_depth = max_depth
                self.min_samples_split = min_samples_split
        ```

        #### `_best_split` Method

        The `_best_split` method is responsible for finding the best feature and threshold 
        combination for splitting the data based on information gain. Let's break down 
        the code step by step:

        - The method starts by initializing variables to keep track of the best feature, 
        threshold, left and right data subsets, and the information gain.

        ```python
        def _best_split(self, data, n_samples, n_features):
            feature = None
            threshold = None
            left = None
            right = None
            gain = -1.0
        ```

        - It then iterates through the features (columns) of the dataset, represented by 
        the variable `col`. For each feature, it calculates the unique thresholds present 
        in that feature using `np.unique(data[:, col])`.

        ```python
        for col in range(n_features):
            thresholds = np.unique(data[:, col])
            for curr_threshold in thresholds:
        ```

        - Within the threshold loop, it creates two lists, `curr_left` and `curr_right`, to 
        collect data points that satisfy and do not satisfy the current threshold condition.

        ```python
        curr_left = []
        curr_right = []
        ```

        - It then iterates through the rows of the data, represented by the variable `row`, 
        and assigns each data point to either `curr_left` or `curr_right` based on whether it 
        satisfies the threshold condition.

        ```python
        for row in range(n_samples):
            if data[row, col] <= curr_threshold:
                curr_left.append(data[row])
            else:
                curr_right.append(data[row])
        ```

        - The method checks if both `curr_left` and `curr_right` have at least one data point 
        (i.e., their lengths are not zero) to ensure that there is a valid split. It then calculates 
        information gain (`curr_gain`) by calling the `information_gain` function with appropriate 
        arguments. Here, information gain is computed considering the variance of `y`, `y_left` and
        `y_right`.

        ```python
        if len(curr_left) != 0 and len(curr_right) != 0:
            y = data[:, -1]
            y_left = curr_left[:, -1]
            y_right = curr_right[:, -1]
            curr_gain = information_gain(y, y_left, y_right, criterion='variance')
        ```

        - If the current gain is greater than the previously recorded best gain, the method 
        updates the variables for the best feature, threshold, and data subsets.

        ```python
        if curr_gain > gain:
            feature = col
            threshold = curr_threshold
            left = curr_left
            right = curr_right
            gain = curr_gain
        ```

        - Finally, the method returns the best feature, threshold, left and right data subsets, 
        and the associated gain.

        ```python
        return feature, threshold, left, right, gain
        ```

        #### `_build` Method

        The `_build` method is responsible for constructing the decision tree. Here's a breakdown 
        of the code:

        - The method starts by extracting the feature matrix `X` and output vector `y` from the input 
        data. It then calculates the number of samples (`n_samples`) and the number of features (`n_features`) 
        in the feature matrix.

        ```python
        def _build(self, data, depth=0):
            X, y = data[:, :-1], data[:, -1]
            n_samples, n_features = X.shape
        ```

        - The method checks two conditions to determine whether further splitting is allowed:
            - `n_samples` must be greater than or equal to the `min_samples_split` hyperparameter.
            - The current depth (`depth`) must be less than or equal to the `max_depth` hyperparameter.
        - If both conditions are met, the method proceeds with further splitting:
            - It calls the `_best_split` method to find the best feature and threshold for the 
            current data.
            - It then recursively calls `_build` on the left and right data subsets (`left` and `right`) 
            and increments the depth by one.
            - A new `Node` is created with the best feature, threshold, left and right child nodes, 
            and the associated gain.

        ```python
        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            feature, threshold, left, right, gain = self._best_split(data, n_samples, n_features)
            left = self._build(left, depth=depth + 1)
            right = self._build(right, depth=depth + 1)
            return Node(feature=feature, threshold=threshold, left=left, right=right, gain=gain)
        ```

        - If the stopping conditions are not met, indicating that further splitting is not allowed, 
        the method creates a leaf node. It computes the mean output value in the current node's data 
        using `np.mean(y)` and assigns it as the value of the leaf node.

        ```python
        else:
            return Node(value=np.mean(y))
        ```

        #### `fit` Method

        The `fit` method is used for training the decision tree on a dataset. Here's how it works:

        - The `fit` method takes two parameters: `X` (features) and `y` (output).
        - It assigns the result of the `_build` method to `self.root`. This is where the decision 
        tree construction process starts.
        - It calls `_build` with the entire dataset, created by stacking the feature matrix `X` and 
        label vector `y` horizontally using `np.column_stack((X, y))`.

        ```python
        def fit(self, X, y):
            self.root = self._build(np.column_stack((X, y)))
        ```

        #### `predict` Method

        The `predict` method is used to make predictions for a dataset. Here's how it's implemented:

        - The `predict` method takes one parameter: `X`, which is the dataset for which predictions 
        are to be made.
        - It uses a list comprehension to make predictions for each data point in `X`. For each data 
        point (row) in `X`, it calls the `make_prediction` method and passes the data point and the root 
        node of the decision tree as arguments.
        - The method returns a list of predicted class labels for each data point in `X`.

        ```python
        def predict(self, X):
            return [self.make_prediction(row, self.root) for row in X.reshape((len(X), 1))]
        ```

        #### `make_prediction` Method

        The `make_prediction` method is a recursive function used to predict the output value of a single 
        data point. Here's how it works:

        - The `make_prediction` method takes two parameters: `x` (the data point to predict) and `tree` 
        (the current node being considered in the decision tree).
        - It first checks if the current node is a leaf node by examining the value attribute. If 
        the value is not `None`, it means it's a leaf node, and that value is returned as the 
        prediction.
        - If the current node is not a leaf node, it compares the feature value of the data point 
        `x` to the threshold in the current node. Depending on the comparison, it recursively calls 
        itself on either the left or right child node until it reaches a leaf node.

        ```python
        def make_prediction(self, x, tree):
            if tree.value != None: 
                return tree.value
            value = x[tree.feature]
            if value <= tree.threshold:
                return self.make_prediction(x, tree.left)
            else:
                return self.make_prediction(x, tree.right)
        ```
        """,
        unsafe_allow_html=True
    )
elif option == KNN_REGRESSION:
    st.markdown(
        """
        ---

        ## $$k$$-Nearest Neighbors (Regression) <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/neighbors.py" style="font-size: 15px">[source]</a>

        The $$k$$-nearest neighbors ($$k$$NN) algorithm is a simple and effective supervised learning 
        algorithm used for classification and regression. It can predict the $$y$$ value for a particular
        data point based on the interpolation of its $$k$$-nearest neighbors.
        
        The following code defines the class required to implement $$k$$NN. This documentation will 
        provide an overview of how the proposed implementation works and describe the key components 
        of the code.

        ### `KNeighborsRegressor` Class

        The `KNeighborsRegressor` class implements the kNN algorithm for regression. Let's go 
        through the key components of the code.

        #### Constructor

        Initializes the $$k$$NN classifier with the number of neighbors to consider (`n_neighbors`).

        ```python
        class KNeighborsRegressor():
            def __init__(self, n_neighbors=5):
                self.n_neighbors = n_neighbors
                self.X_train = None
                self.y_train = None
        ```

        #### `fit` Method

        Fits the $$k$$NN model to the provided training data. This algorithm is considered to
        be a *lazy* algorithm, meaning that all computation occurs at test time.

        ```python
        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
        ```

        #### `predict` Method

        Predicts a $$y$$ value for the given data points.
        
        - This method first computes the Euclidean distance between each point in X and all 
        points in the training data:
        
        ```python
        def predict(self, X):
            dist = np.array([[np.linalg.norm(p - q) for q in self.X_train.reshape((len(self.X_train), 1))] for p in X.reshape((len(X), 1))])
        ```

        - It then identifies the indices of the $$k$$-nearest neighbors for each point:

        ```python
        neighbors_idx = np.argpartition(dist, self.n_neighbors)[:, :self.n_neighbors]
        ```
        
        - Lastly, it predicts the output value by computing the mean $$y$$ coordinate of 
        the considered neighbors:

        ```python
        y_pred = np.mean(self.y_train[neighbors_idx], axis=1).tolist()
        ```
        """,
        unsafe_allow_html=True
    )
elif option == KMEANS:
    st.markdown(
        """
        ---

        ## K-Means <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/unsupervised/cluster.py" style="font-size: 15px">[source]</a>

        The `KMeans` class in Python is an implementation of the K-means clustering algorithm, a widely-used 
        method in unsupervised learning for grouping data into a pre-defined number of clusters. Each 
        cluster in K-means clustering is defined by its center, which is typically the mean of the points 
        in the cluster.

        ### `KMeans` Class

        The `KMeans` class is used for partitioning a dataset into $$k$$ distinct, non-overlapping subgroups 
        (clusters) where each data point belongs to the cluster with the nearest mean. It is suitable 
        for scenarios where the number of clusters is known a priori.

        #### Constructor

        The constructor method initializes the K-means clustering model with default settings.

        ```python
        class KMeans():
            def __init__(self, n_clusters=2, n_iter=100):
                self.n_clusters = n_clusters
                self.n_iter = n_iter
                self.centers = None
        ```

        Here, `n_clusters` is the number of clusters to form and the number of centroids to generate.
        `n_iter` is the number of iterations for the K-means clustering algorithm.

        #### `fit` Method

        The `fit` method is used to compute K-means clustering on the provided dataset.

        ```python
        def fit(self, X):
            n_samples = len(X)
            self.centers = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
            for _ in range(self.n_iter):
                clusters = self.predict(X)
                new_centers = np.array([X[clusters == j].mean(axis=0) for j in range(self.n_clusters)])
                if np.all(self.centers == new_centers):
                    break
                self.centers = new_centers
        ```

        The method performs the following steps:
        1. Initializes the cluster centers randomly from the data points.
        2. Assigns each data point to the nearest cluster center.
        3. Recomputes the cluster centers as the mean of the assigned data points.
        4. Repeat steps 2 and 3 for a fixed number of iterations or until cluster centers do not change.

        #### `fit_predict` Method

        This method is a convenience that computes cluster centers and predicts the cluster index for 
        each sample in `X`.

        ```python
        def fit_predict(self, X):
            self.fit(X)
            return self.predict(X)
        ```

        #### predict Method

        After the model is trained (cluster centers are computed), the predict method can be used to 
        assign each sample in `X` to the nearest cluster center. This method calculates the Euclidean 
        distance of each data point from each cluster center and assigns the data point to the closest 
        cluster.

        ```python
        def predict(self, X):
            distances = np.sqrt(((X[:, np.newaxis] - self.centers)**2).sum(axis=2))
            return np.argmin(distances, axis=1)
        ```
        """,
        unsafe_allow_html=True
    )
elif option == DBSCAN:
    st.markdown(
        """
        ---

        ## DBSCAN <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/unsupervised/cluster.py" style="font-size: 15px">[source]</a>

        DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular clustering 
        algorithm used in machine learning. Unlike centroid-based algorithms like K-Means, DBSCAN 
        groups together points that are closely packed together while marking points in low-density 
        regions as outliers or noise. This algorithm excels in finding clusters of arbitrary shapes 
        and sizes, making it suitable for various applications.

        ### `DBSCAN` Class

        This class implements the DBSCAN clustering algorithm.

        #### Constructor

        Initializes the `DBSCAN` object with the specified `eps` and `min_samples` values. It also 
        initializes `self.labels` to None, which will later store the cluster labels for each point.

        ```python
        class DBSCAN():
            def __init__(self, eps=0.5, min_samples=5):
                self.eps = eps
                self.min_samples = min_samples
                self.labels = None
        ```

        Here, `eps` is the maximum distance between two samples for one to be considered as in the 
        neighborhood of the other. `min_samples` is the number of samples in a neighborhood for a 
        point to be considered a core point.

        #### `fit` Method

        Runs the DBSCAN clustering algorithm on dataset `X`. First,

        ```python
        self.labels = [0] * len(X)
        label = 0
        ```

        initializes the `labels` list with zeros for all points in `X`. Zero indicates that a point 
        has not been considered yet. Moreover, it initializes the current cluster label to zero.
        Then,

        ```python
        for i in range(len(X)):
            if not (self.labels[i] == 0):
                continue
            neighbors = self._compute_neighborhood(X, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
        ```

        the `for` loop iterates over all points in `X`. `if not (self.labels[i] == 0): continue` skips 
        points already assigned to a cluster or marked as noise. `neighbors = self._compute_neighborhood(X, i)` 
        finds the neighbors of the current point `i`. `if len(neighbors) < self.min_samples` checks 
        if the point `i` is a noise point. If that is the case, than `-1` is assigned to that point.

        ```python
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
        ```

        Here, the cluster label is incremented, `self.labels[i] = label` assigns the cluster label to 
        point `i`, and the `while` loop iterates through all neighbors of the current point `i`. Inside the 
        loop, each neighbor is checked and, if unvisited or marked as noise, it is added to the current 
        cluster. In the end, `neighbors += new_neighbors` expands the `neighbors` list with new neighbors 
        of the current neighbor point, if it is also a core point.

        #### `fit_predict` Method

        Runs the DBSCAN clustering algorithm on dataset `X` and returns the cluster labels.

        ```python
        def fit_predict(self, X):
            self.fit(X)
            return self.labels
        ```

        #### `_compute_neighborhood` Method

        Computes and returns the neighborhood of a point `p` within distance `eps`.

        ```python
        def _compute_neighborhood(self, X, p):
            neighbors = []
            for i in range(len(X)):
                if np.linalg.norm(X[p] - X[i]) < self.eps:
                    neighbors.append(i)
            return neighbors
        ```

        The method works as follows:
        1. It initializes an empty list `neighbors`.
        2. It iterates over all points in `X`.
        3. It checks if the distance between point `p` and another point `i` is less than `eps`.
        4. It adds the index of point `i` to neighbors if it is within `eps` distance from point `p`.
        """,
        unsafe_allow_html=True
    )