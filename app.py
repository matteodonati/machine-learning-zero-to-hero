import streamlit as st
from st_pages import show_pages_from_config

st.set_page_config(layout='centered')
st.title('Machine Learning: From Zero to Hero :rocket:')
show_pages_from_config()

st.markdown(
    """
    - Supervised Learning
        - <a href='Classification' target='_self'>Classification</a>
        - <a href='Regression' target='_self'>Regression</a>
    - Unsupervised Learning
        - <a href='Clustering' target='_self'>Clustering</a>
    - <a href='Documentation' target='_self'>Documentation</a>
    """, 
    unsafe_allow_html=True
)