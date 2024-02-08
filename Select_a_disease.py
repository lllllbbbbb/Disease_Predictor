import streamlit as st
from streamlit_extras.app_logo import add_logo
st.set_page_config(
    page_title="Better Life Disease Predictor",
    page_icon="⚕️",
)
page_bg_img = """
<style>
[data-testid=stAppViewContainer]{
 background: rgb(255,255,255);
background: linear-gradient(90deg, rgba(255,255,255,1) 0%, rgba(232, 236, 241,1) 100%);
}
[data-testid=stHeader]{
background-color: rgba(0,0,0,0);
}
.css-1aumxhk {
background-color: #011839;
background-image: none;
color: #ffffff
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# add kitten logo
#add_logo("logo_t.png")
with st.sidebar:
  st.image("logo_t.png")
#st.sidebar.image('logo_t.png')

st.title("Welcome to Better Life Diseases Predictor")
st.image('logo_t.png')

st.header("<--- Please try our predictor by clicking on a page on the left.")