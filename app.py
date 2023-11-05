import streamlit as st
from sklearn.datasets import make_moons
from ml.utils.data import train_test_split
from ui.utils.plot import plot_data

st.set_page_config(layout='wide')
st.title('Machine Learning: From Zero to Hero :rocket:')

X, y = make_moons(n_samples=500, noise=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

col1, col2, col3 = st.columns(3)
with col1:
    tab_train, tab_test = st.tabs(['Train', 'Test'])
    with tab_train:
        fig_train = plot_data(X_train, y_train, 'Train', marker_symbol='circle')
        st.plotly_chart(fig_train, use_container_width=True)
    with tab_test:
        fig_test = plot_data(X_test, y_test, 'Test', marker_symbol='square')
        st.plotly_chart(fig_test, use_container_width=True)