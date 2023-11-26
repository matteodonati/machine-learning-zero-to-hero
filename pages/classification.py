import streamlit as st
from sklearn.datasets import make_moons, make_blobs
from ml.utils.data import train_test_split, normalize_data
from ml.utils.metrics import accuracy_score, precision_score, recall_score
from ui.utils.plot import create_plot, add_data_to_plot, add_decision_boundary
from ml.supervised.classification.tree import DecisionTreeClassifier
from ml.supervised.classification.naive_bayes import GaussianNB
from ml.supervised.classification.neighbors import KNeighborsClassifier
from ml.supervised.classification.linear import LogisticRegression

st.set_page_config(layout='wide')
st.title('Classification :bar_chart:')

st.header('Data')
col_data, col_data_info = st.columns(2, gap='large')

with col_data:

    MOONS = 'Moons'
    BLOBS = 'Blobs'

    data_option = st.selectbox(
        'Select a dataset',
        (MOONS, BLOBS),
    )

    n_samples = st.slider('Select the number of samples', 0, 1000, 500, step=10)
    if data_option == MOONS:
        noise = st.slider('Select the amount of noise', 0.0, 0.1, 0.05)
        X, y = make_moons(n_samples=n_samples, noise=noise)
    elif data_option == BLOBS:
        centers = st.slider('Select the number of clusters', 0, 5, 2)
        X, y = make_blobs(n_samples=n_samples, centers=centers)

    normalize_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

with col_data_info:
    st.markdown(
        '''
        Data info
        - Data info
        - Data info
        '''
    )

st.header('Model')
col_model, col_model_info = st.columns(2, gap='large')

with col_model:

    DECISION_TREE = 'Decision tree'
    NAIVE_BAYES = 'Gaussian naive Bayes'
    KNN = 'K-nearest neighbors'
    LR = 'Logistic regression'

    model_option = st.selectbox(
        'Select a model',
        (DECISION_TREE, NAIVE_BAYES, KNN, LR),
    )
    if model_option == DECISION_TREE:
        max_depth = st.slider('Select the maximum depth', 0, 10, 5)
        min_samples_split = st.slider('Select the minimum number of samples required to split a node', 0, 5, 2)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    elif model_option == NAIVE_BAYES:
        model = GaussianNB()
    elif model_option == KNN:
        n_neighbors = st.slider('Select the number of neighbors', 1, 10, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_option == LR:
        n_epochs = st.slider('Select the number of training epochs', 50, 150, 100)
        lr = st.slider('Select the learning rate', 0.0001, 0.01, 0.001)
        model = LogisticRegression(n_epochs=n_epochs, lr=lr)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

with col_model_info:
    st.markdown(
        '''
        Model info
        - Model info
        - Model info
        '''
    )

st.header('Results')
col_results, col_results_info = st.columns(2, gap='large')

with col_results:
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

with col_results_info:
    st.markdown(
        '''
        Results info
        - Results info
        - Results info
        '''
    )