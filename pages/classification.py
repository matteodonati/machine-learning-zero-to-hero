import streamlit as st
from sklearn.datasets import make_moons, make_blobs, make_circles
from ml.utils.data import train_test_split, normalize_data
from ml.utils.metrics import accuracy_score, precision_score, recall_score
from ui.utils.plot import create_plot, add_data_to_plot, add_decision_boundary
from ml.supervised.tree import DecisionTreeClassifier
from ml.supervised.naive_bayes import GaussianNB
from ml.supervised.neighbors import KNeighborsClassifier
from ml.supervised.linear import LogisticRegression
from ml.supervised.svm import SVC

st.title('Classification :bar_chart:')

st.markdown(
    """
    This page showcases the implementation of the classification Tool. 
    With this tool, you have the flexibility to choose from three default 
    datasets, select different models, and visualize the predicted outcomes. 
    Additionally, you can tailor the data parameters and fine-tune the 
    model hyperparameters to study the behavior of the chosen classifier.
    """
)

MOONS = 'Moons'
BLOBS = 'Blobs'
CIRCLES = 'Circles'
CLASSIFICATION_TREE = 'Classification tree'
NAIVE_BAYES = 'Gaussian naive Bayes'
KNN = 'K-nearest neighbors'
LR = 'Logistic regression'
SVM = 'Support vector machine'

st.header('Data')

data_option = st.selectbox(
    'Select a dataset',
    (MOONS, BLOBS, CIRCLES),
)

n_samples = st.slider('Select the number of samples', 0, 1000, 500, step=10)
if data_option == MOONS:
    noise = st.slider('Select the amount of noise', 0.0, 0.1, 0.05)
    X, y = make_moons(n_samples=n_samples, noise=noise)
elif data_option == BLOBS:
    centers = st.slider('Select the number of clusters', 0, 5, 2)
    X, y = make_blobs(n_samples=n_samples, centers=centers)
elif data_option == CIRCLES:
    noise = st.slider('Select the amount of noise', 0.0, 0.1, 0.05)
    X, y = make_circles(n_samples=n_samples, noise=noise)

normalize_data(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

st.header('Model')

model_option = st.selectbox(
    'Select a model',
    (CLASSIFICATION_TREE, NAIVE_BAYES, KNN, LR, SVM),
)

if model_option == CLASSIFICATION_TREE:
    max_depth = st.slider('Select the maximum depth', 0, 10, 5)
    min_samples_split = st.slider('Select the minimum number of samples required to split a node', 0, 5, 2)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
elif model_option == NAIVE_BAYES:
    model = GaussianNB()
elif model_option == KNN:
    n_neighbors = st.slider('Select the number of neighbors', 1, 10, 5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
elif model_option == LR:
    n_epochs = st.slider('Select the number of training iterations', 50, 150, 100)
    lr = st.slider('Select the learning rate value', min_value=0.001, step=0.001, max_value=0.01, value=0.005, format='%f')
    model = LogisticRegression(n_epochs=n_epochs, lr=lr)
elif model_option == SVM:
    n_epochs = st.slider('Select the number of training iterations', 500, 1500, 1000)
    lr = st.slider('Select the learning rate value', min_value=0.01, step=0.01, max_value=0.1, value=0.05, format='%f')
    model = SVC(lr=lr, n_epochs=n_epochs)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.header('Results')

if data_option == BLOBS and centers > 2 and (model_option == LR or model_option == SVM):
    st.info(f'{model_option} can\'t solve classification problems with more than two classes.', icon='ℹ️')
else:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    st.progress(accuracy, text='Test accuracy: {:.6f}'.format(accuracy))
    st.progress(precision, text='Test precision: {:.6f}'.format(precision))
    st.progress(recall, text='Test recall: {:.6f}'.format(recall))

    fig_train = create_plot()
    fig_test = create_plot()
    add_data_to_plot(fig_train, X_train, y_train, marker_symbol='circle')
    add_data_to_plot(fig_test, X_test, y_test, marker_symbol='square')
    add_decision_boundary(fig_train, fig_test, X, model)

    tab_train, tab_test = st.tabs(['Train', 'Test'])
    with tab_train:
        st.plotly_chart(fig_train, use_container_width=True)
    with tab_test:        
        st.plotly_chart(fig_test, use_container_width=True)