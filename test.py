import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_regression
from ml.utils.data import train_test_split, make_sin
from ml.utils.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from ml.supervised.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ml.supervised.naive_bayes import GaussianNB
from ml.supervised.neighbors import KNeighborsClassifier, KNeighborsRegressor
from ml.supervised.linear import LogisticRegression, LinearRegression
from ml.supervised.svm import SVC

np.random.seed(0)

print('Downloading data.')
X, y = make_sin(n_samples=500)
X = X[:, 0]
#X, y = make_blobs(centers=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print('Fitting the model to train data.')
#model = DecisionTreeClassifier(max_depth=10)
#model = GaussianNB()
#model = KNeighborsClassifier()
#model = LogisticRegression()
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = KNeighborsRegressor()
model.fit(X_train, y_train)

print('Predicting values for test data.')
y_pred = model.predict(X_test)

#print(accuracy_score(y_test, y_pred))
#print(precision_score(y_test, y_pred))
#print(recall_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))