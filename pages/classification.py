import streamlit as st
from sklearn.datasets import make_moons, make_blobs
from ml.utils.data import train_test_split
from ui.utils.plot import create_plot, add_data_to_plot
from ml.supervised.classification.tree import DecisionTreeClassifier
from ml.utils.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(layout='wide')
st.title('Classification Playground :bar_chart:')

st.header('Data')
col_data, col_data_info = st.columns(2, gap='large')

with col_data:
    data_option = st.selectbox(
        'Select a dataset',
        ('Moons', 'Blobs'),
    )
    n_samples = st.slider('Select the number of samples', 0, 1000, 500, step=10)
    if data_option == 'Moons':
        noise = st.slider('Select the amount of noise', 0.0, 0.1, 0.05)
        X, y = make_moons(n_samples=n_samples, noise=noise)
    elif data_option == 'Blobs':
        centers = st.slider('Select the number of clusters', 0, 5, 2)
        X, y = make_blobs(n_samples=n_samples, centers=centers)
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
    model_option = st.selectbox(
        'Select a model',
        ('Decision tree',),
    )
    max_depth = st.slider('Select the maximum depth', 0, 5, 2)
    min_samples_split = st.slider('Select the minimum number of samples required to split a node', 0, 5, 2)
    if model_option == 'Decision tree':
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
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
    st.progress(accuracy, text='Accuracy: {:.2f}'.format(accuracy))
    st.progress(precision, text='Precision: {:.2f}'.format(precision))
    st.progress(recall, text='Recall: {:.2f}'.format(recall))

    tab_train, tab_test = st.tabs(['Train', 'Test'])
    with tab_train:
        fig_train = create_plot()
        add_data_to_plot(fig_train, X_train, y_train, marker_symbol='circle')
        st.plotly_chart(fig_train, use_container_width=True)
    with tab_test:
        fig_test = create_plot()
        add_data_to_plot(fig_test, X_test, y_test, marker_symbol='square')
        st.plotly_chart(fig_test, use_container_width=True)

with col_results_info:
    st.markdown(
        '''
        Results info
        - Results info
        - Results info
        '''
    )