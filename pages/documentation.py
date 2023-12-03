import streamlit as st

st.set_page_config(layout='centered')
st.title('Documentation :page_facing_up:')

DECISION_TREE = 'Decition tree'
NAIVE_BAYES = 'Gaussian naive Bayes'
KNN = 'K-nearest neighbors'
LR = 'Logistic regression'
SVM = 'Support vector machine'

option = st.selectbox(
    'Select a documentation page',
    (DECISION_TREE, NAIVE_BAYES, KNN, LR, SVM),
    index=None,
)

st.markdown(
    """
    <script
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
        type="text/javascript">
    </script>
    """,
    unsafe_allow_html=True
)

if option == DECISION_TREE:
    st.markdown(
        """
        ---
        
        ## Classification Tree <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/classification/tree.py" style="font-size: 15px">[source]</a>

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
        
        ## Gaussian Naive Bayes <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/classification/naive_bayes.py" style="font-size: 15px">[source]</a>
        
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

        ## $$k$$-Nearest Neighbors <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/classification/neighbors.py" style="font-size: 15px">[source]</a>

        The $$k$$-nearest neighbors ($$k$$NN) algorithm is a simple and effective supervised learning 
        algorithm used for classification and regression. It classifies a data point based on 
        the majority class among its $$k$$-nearest neighbors. The following code defines the class
        required to implement $$k$$NN. This documentation will provide an overview of how the proposed 
        implementation works and describe the key components of the code.

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

        ## Logistic Regression <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/classification/linear.py" style="font-size: 15px">[source]</a>
        """,
        unsafe_allow_html=True
    )
elif option == SVM:
    st.markdown(
        """
        ---

        ## Support Vector Machine <a href="https://github.com/matteodonati/machine-learning-zero-to-hero/blob/main/ml/supervised/classification/svm.py" style="font-size: 15px">[source]</a>
        """,
        unsafe_allow_html=True
    )
