import streamlit as st
import base64
from st_pages import show_pages_from_config
from pathlib import Path

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

with st.sidebar:
    img = Path('./images/github.png').read_bytes()
    encoded = base64.b64encode(img).decode()
    st.markdown(
        """
        [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/matteodonati/machine-learning-zero-to-hero) 
        <small> &nbsp; Machine Learning: From Zero to Hero </small>
        """.format(encoded),
        unsafe_allow_html=True,
    )