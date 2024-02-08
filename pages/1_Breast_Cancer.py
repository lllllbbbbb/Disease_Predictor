import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import RobustScaler
from streamlit_extras.let_it_rain import rain

with st.sidebar:
  st.image("logo_t.png")

page_bg_img = """
<style>
[data-testid=stAppViewContainer]{
 background: rgb(255,255,255);
background: linear-gradient(90deg, rgba(255,255,255,1) 0%, rgba(232, 236, 241,1) 100%);
}
[data-testid=stHeader]{
background-color: rgba(0,0,0,0);
}
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #31333f;
    color:#FFFFFF;
}
div.stButton > button:hover {
    background-color: #FFFFFF;
    color:#000000;
    }
</style>""", unsafe_allow_html=True)
# imports
@st.cache_data
def load_data():
    voting_cancer = load('voting_cancer.joblib')
    df_cancer = pd.read_parquet('df_breast_cancer.parquet')
    scaler_cancer = load('scaler_cancer.joblib')
    return(voting_cancer, df_cancer, scaler_cancer)
voting_cancer, df_cancer, scaler_cancer = load_data()
#
st.title('Breast Cancer Predictor')
st.header('Please enter your medical data')
col0, col1, col2 = st.columns([1,1,3])
with col0:
    tx = st.number_input(f"**Texture** ", value=18.85, min_value=0.0)
    ar = st.number_input(f"**Area (µm^2)**", value=557.65, min_value=0.0)
    smt = st.number_input(f"**Smoothness** ", value=0.1, min_value=0.0)
    cn = st.number_input(f"**Compactness** ", value=0.1, min_value=0.0)
with col1:
    cv = st.number_input(f"**Concavity** ", value=0.1, min_value=0.0)
    cp = st.number_input(f"**Concave Points** ", value=0.03, min_value=0.0)
    sm = st.number_input(f"**Symmetry** ", value=0.18, min_value=0.0)
    fd = st.number_input(f"**Fractal Dimension** ", value=0.06, min_value=0.0)

st.divider()
if st.button(r"$\textsf{\Huge Discover the risk}$"):
    scaled_udf = scaler_cancer.transform(np.array([[tx,ar,smt,cn,cv,cp,sm,fd]]))
    outcome = voting_cancer.predict(scaled_udf)
    certainity = voting_cancer.predict_proba(scaled_udf)
    if outcome[0] == 0:
        st.success("The cell might be benign.")
        st.write("Certainity :",round(certainity[0][0]*100,2),"%")

    else:
        st.error("The cell might be malignant. Please contact your general practitioner.")
        st.write("Certainity :",round(certainity[0][1]*100,2),"%")

st.info("The predictions on this site are provided for information purposes only. The content is in no way intended as a substitute for a medical examination, consultation, diagnosis or treatment")
