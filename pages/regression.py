import streamlit as st
from sklearn.datasets import make_regression
from ml.utils.data import train_test_split, make_sin
from ui.utils.plot import create_plot, add_data_to_plot, add_regression_line
from ml.supervised.linear import LinearRegression
from ml.supervised.tree import DecisionTreeRegressor
from ml.supervised.neighbors import KNeighborsRegressor

st.set_page_config(layout='wide')
st.title('Regression :chart_with_upwards_trend:')

st.header('Data')
col_data, col_data_info = st.columns(2, gap='large')

with col_data:

    REGRESSION = 'Linear'
    SINE = 'Sine'

    data_option = st.selectbox(
        'Select a dataset',
        (REGRESSION, SINE),
    )

    n_samples = st.slider('Select the number of samples', 0, 1000, 500, step=10)
    if data_option == REGRESSION:
        noise = st.slider('Select the amount of noise', 0.0, 50.0, 25.0)
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise)
    elif data_option == SINE:
        noise = st.slider('Select the amount of noise', 0.0, 1.0, 0.3)
        X, y = make_sin(n_samples=n_samples, noise=noise)
    
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

    LINEAR = 'Linear regression'
    REGRESSION_TREE = 'Regression tree'
    KNN = 'K-nearest neighbors'

    model_option = st.selectbox(
        'Select a model',
        (LINEAR, REGRESSION_TREE, KNN),
    )

    if model_option == LINEAR:
        n_epochs = st.slider('Select the number of training iterations', 1000, 3000, 2000)
        lr = st.slider('Select the learning rate value', min_value=0.0001, step=0.001, max_value=0.005, value=0.001, format='%f')
        model = LinearRegression(n_epochs=n_epochs, lr=lr)
    elif model_option == REGRESSION_TREE:
        max_depth = st.slider('Select the maximum depth', 0, 10, 5)
        min_samples_split = st.slider('Select the minimum number of samples required to split a node', 0, 5, 2)
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    elif model_option == KNN:
        n_neighbors = st.slider('Select the number of neighbors', 1, 10, 5)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    
    X_train = X_train[:, 0]
    model.fit(X_train, y_train)

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
    fig = create_plot()
    add_data_to_plot(fig, X_train, y_train, marker_symbol='circle', problem_type='regression')
    add_regression_line(fig, X, model)
    st.plotly_chart(fig, use_container_width=True)

with col_results_info:
    st.markdown(
        '''
        Results info
        - Results info
        - Results info
        '''
    )