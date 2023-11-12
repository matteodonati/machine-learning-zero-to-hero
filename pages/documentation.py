import streamlit as st

st.set_page_config(layout='centered')
st.title('Documentation :page_facing_up:')

CLASSIFICATION_TREE = 'Classification tree'

option = st.selectbox(
    'Select a documentation page',
    (CLASSIFICATION_TREE,),
    index=None,
)

if option == CLASSIFICATION_TREE:
    st.markdown(
        """
        ---
        
        ## Classification Tree

        The following code defines the classes required to implement a decision 
        tree model for classification. Decision trees are a popular machine 
        learning algorithm used for both classification and regression tasks. 
        This documentation will provide an overview of how the proposed implementation 
        works and describe the key components of the code.

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
        """
    )