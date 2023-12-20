import streamlit as st
from sklearn.datasets import make_moons, make_blobs, make_circles
from ui.utils.plot import create_plot, add_clustering_scheme
from ml.utils.data import normalize_data
from ml.unsupervised.cluster import KMeans, DBSCAN

st.set_page_config(layout='wide')
st.title('Clustering :white_square_button:')

st.header('Data')
col_data, col_data_info = st.columns(2, gap='large')

with col_data:

    MOONS = 'Moons'
    BLOBS = 'Blobs'
    CIRCLES = 'Circles'

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
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.3)

    normalize_data(X)

with col_data_info:
    st.markdown(
        '''
        Data info
        - Data info
        - Data info
        '''
    )

st.header('Algorithm')
col_algo, col_algo_info = st.columns(2, gap='large')

with col_algo:

    K_MEANS = 'K-means'
    _DBSCAN = 'DBSCAN'

    algo_option = st.selectbox(
        'Select an algorithm',
        (K_MEANS, _DBSCAN),
    )
    
    # TODO
    if algo_option == K_MEANS:
        n_clusters = st.slider('Select the number of clusters', 1, 5, 2)
        n_iter = st.slider('Select the number of iterations', 100, 300, 200)
        algo = KMeans(n_clusters=n_clusters, n_iter=n_iter)
    elif algo_option == _DBSCAN:
        eps = st.slider('Select the epsilon parameter', 0.1, 0.2, 0.15)
        min_samples = st.slider('Select the number of samples in a neighborhood for a point to be considered as a core point', 1, 10, 5)
        algo = DBSCAN(eps=eps, min_samples=min_samples)

    y_pred = algo.fit_predict(X)

with col_algo_info:
    st.markdown(
        '''
        Algorithm info
        - Algorithm info
        - Algorithm info
        '''
    )

st.header('Results')
col_results, col_results_info = st.columns(2, gap='large')

with col_results:
    fig = create_plot()
    add_clustering_scheme(fig, X[:, 0], X[:, 1], y_pred)
    st.plotly_chart(fig, use_container_width=True)

with col_results_info:
    st.markdown(
        '''
        Results info
        - Results info
        - Results info
        '''
    )