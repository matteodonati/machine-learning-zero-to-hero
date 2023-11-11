import streamlit as st

st.set_page_config(layout='centered')
st.title('Documentation :page_facing_up:')

option = st.selectbox(
    'Select a documentation page',
    ('Classification tree',),
    index=None,
)