import numpy as np
from sklearn.datasets import make_moons
from ml.utils.data import train_test_split
from ml.utils.metrics import accuracy_score, precision_score, recall_score
from ml.supervised.classification.tree import DecisionTreeClassifier

np.random.seed(0)

print('Downloading data.')
X, y = make_moons()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train.shape, X_test.shape)

print('Fitting the model to train data.')
model = DecisionTreeClassifier(max_depth=10)
model.fit(X_train, y_train)

print('Predicting labels for test data.')
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))