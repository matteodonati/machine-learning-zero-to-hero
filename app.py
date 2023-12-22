import streamlit as st
from st_pages import show_pages_from_config

st.set_page_config(layout='centered')
st.title('Machine Learning: From Zero to Hero :rocket:')
show_pages_from_config()

st.markdown(
    """
    Welcome to the Machine Learning: From Zero to Hero! This web app is a 
    versatile tool that empowers you to explore various aspects of machine 
    learning. Whether you are interested in classification, regression, 
    or clustering, this application has got you covered. All the models 
    and algorithms implemented in this web app are custom-built, and exhibit 
    an interface that closely resembles that of scikit-learn.

    ## Key Features

    ### Classification

    The classification tool allows you to apply machine learning models to 
    classify data with ease. You can pick a default dataset, configure the 
    model settings, and witness the power of intelligent classification. 
    The results are presented vividly through interactive plots, providing 
    valuable insights into the data, and models' capabilities.

    ### Regression

    With the regression tool, you can delve into the world of predictive 
    analytics. Regression algorithms give you the ability to make accurate 
    predictions. Whether you are tackling linear or nonlinear relationships, 
    you can visualize the results using informative plots.

    ### Clustering

    The clustering tool provides a deeper understanding of the data's inherent 
    structure. Utilize clustering algorithms to group similar data points 
    together. The interactive visualizations enable you to grasp the patterns 
    and relationships within your data, and better understand the pros and
    cons of some of the most famous clustering algorithms.

    ### Documentation

    To aid your understanding, I have included comprehensive documentation for 
    each model and algorithm implemented in this web app. You can explore the 
    intricacies of the methods used, gaining insights into the underlying 
    principles of machine learning.

    ## Get Started

    Get started by selecting one of the following links, which will direct you 
    to the array of available tools. You can also explore the web app using 
    the left-hand sidebar for navigation.
    
    - Supervised Learning
        - <a href='Classification' target='_self'>Classification</a>
        - <a href='Regression' target='_self'>Regression</a>
    - Unsupervised Learning
        - <a href='Clustering' target='_self'>Clustering</a>
    - <a href='Documentation' target='_self'>Documentation</a>

    ## Contribute

    Contributions from the community are welcome! If you are passionate about 
    machine learning and would like to contribute, you can do it via the [project's 
    GitHub repository](https://github.com/matteodonati/machine-learning-zero-to-hero).
    """, 
    unsafe_allow_html=True
)